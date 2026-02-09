import sys
import asyncio
from datetime import datetime, timezone
from PyQt5 import QtWidgets, uic
from qasync import QEventLoop, asyncSlot
from tle_loader import save_tle, dataset_sat, parse_tle_fields
from sgp4_core import simulate

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("main.ui", self)

        # --- AI ---
        self.sendAI.clicked.connect(self.on_send)
        self.quest.returnPressed.connect(self.on_send)
        self.Answ.setReadOnly(True)

        # --- Console / Commands ---
        self.consoleTextEdit.setReadOnly(True)
        self.command_buffer = []
        self.sendCommandButton.clicked.connect(self.add_command_to_console)
        self.sendPackButton.clicked.connect(self.send_command_pack)
        self.commandLineEdit.returnPressed.connect(self.add_command_to_console)

        # --- Satellite loading state ---
        self.current_sat_number = None
        self.loading_stage = None
        self.sat_data = {}

        # --- SGP4 ---
        self.consoleTextEdit.append("[SYSTEM] Console ready\n")
        save_tle()
        self.consoleTextEdit.append("✔ TLE database loaded")

    # --- AI Slot ---
    @asyncSlot()
    async def on_send(self):
        user_text = self.quest.text().strip()
        if not user_text:
            return
        self.Answ.append(f"<b>Вы:</b> {user_text}")
        self.quest.clear()

        try:
            name_line, line1, line2 = split_tle_line(user_text)
        except Exception as e:
            self.Answ.append(f"[Ошибка разбора TLE]: {e}")
            return

        # Отправляем на сервер
        response = await ai_answer({"tle_lines": [name_line, line1, line2]})

        # Просто выводим результат
        self.Answ.append(f"<b>ИИ:</b> {response}")

    # --- Command Handling ---
    def add_command_to_console(self):
        command = self.commandLineEdit.text().strip()
        if not command:
            return

        self.consoleTextEdit.append(f"> {command}")
        self.commandLineEdit.clear()

        # ЕСЛИ вводим параметры спутника — ОБРАБАТЫВАЕМ И ВЫХОДИМ
        if self.loading_stage is not None:
            self.process_satellite_parameter(command)
            return

        parts = command.split()

        # ===== SIMULATE COMMAND =====
        if parts[0].lower() == "simulate":
            try:
                duration_hours = float(parts[1])
                step_minutes = float(parts[2])
            except (IndexError, ValueError):
                self.consoleTextEdit.append(
                    "Usage: simulate <hours> <step_minutes>\n"
                )
                return

            self.run_simulation(duration_hours, step_minutes)
            return

        # ===== SAT COMMAND =====
        if parts[0].lower() == "sat":
            try:
                sat_number = int(parts[1])
                self.start_satellite_loading(sat_number)
            except (IndexError, ValueError):
                self.consoleTextEdit.append("❌ Invalid satellite number")
            return

        # Обычная команда
        self.command_buffer.append(command)

    def start_satellite_loading(self, sat_number):
        """Начинает процесс загрузки спутника"""
        try:
            # Загружаем TLE данные
            number_sat = (sat_number - 1) * 3
            line0, line1, line2 = dataset_sat(number_sat)

            self.consoleTextEdit.append(f"✔ Satellite found: {line0}")
            self.consoleTextEdit.append(f"   {line1}")
            self.consoleTextEdit.append(f"   {line2}\n")

            # Сохраняем данные спутника
            self.sat_data = {
                'sat_number': sat_number,
                'line0': line0,
                'line1': line1,
                'line2': line2,
                'deltas': {}
            }

            # Парсим поля TLE для отображения
            fields = parse_tle_fields(line1, line2)

            # Заполняем базовую телеметрию
            self.name_value.setText(line0)
            self.num_value.setText(fields['satellite_number'])
            self.designator_value.setText(fields['international_designator'])
            self.inclination_value.setText(fields['inclination'])
            self.ascendation_value.setText(fields['raan'])
            self.date_time_value.setText(f"{fields['epoch_year']} {fields['epoch_day']}")
            self.eccentricity_value.setText(fields['eccentricity'])
            self.perrige_value.setText(fields['argument_perigee'])
            self.ballistic_value.setText(fields['bstar'])
            self.dirrative_value.setText(fields['second_derivative'])
            self.anomaly_value.setText(fields['mean_anomaly'])
            self.pressure_value.setText(fields['bstar'])  # или другое поле
            self.mean_motion_value.setText(fields['mean_motion'])
            self.ephemeris_value.setText(fields['ephemeris_type'])
            self.element_value.setText(fields['element_number'])
            self.rottation_value.setText(fields['revolution_number'])

            # Начинаем запрос дельт
            self.loading_stage = 'inclination'
            self.consoleTextEdit.append("[INPUT] Enter DELTA values for orbit modification:")
            self.consoleTextEdit.append("DELTA Inclination:")
        except Exception as e:
            self.consoleTextEdit.append(f"❌ Error loading satellite: {e}")
            self.loading_stage = None

    def process_satellite_parameter(self, value):
        """Обрабатывает ввод параметров спутника"""
        try:
            param_value = float(value)
            # Преобразуем в число
            # Сохраняем дельту
            if self.loading_stage == 'inclination':
                self.sat_data['deltas']['inclination'] = param_value
                self.loading_stage = 'raan'
                self.consoleTextEdit.append("DELTA RAAN:")

            elif self.loading_stage == 'raan':
                self.sat_data['deltas']['raan'] = param_value
                self.loading_stage = 'eccentricity'
                self.consoleTextEdit.append("DELTA Eccentricity:")

            elif self.loading_stage == 'eccentricity':
                self.sat_data['deltas']['eccentricity'] = param_value
                self.loading_stage = 'argument_perigee'
                self.consoleTextEdit.append("DELTA Argument_of_Perigee:")

            elif self.loading_stage == 'argument_perigee':
                self.sat_data['deltas']['argument_perigee'] = param_value
                self.loading_stage = 'mean_motion'
                self.consoleTextEdit.append("DELTA Mean_Motion:")


            elif self.loading_stage == 'mean_motion':
                self.sat_data['deltas']['mean_motion'] = param_value
                self.loading_stage = None
                self.consoleTextEdit.append("✔ DELTA input completed\n")  # ✅ только здесь
                # Применяем дельты и обновляем телеметрию
                self.apply_deltas_and_update()


        except ValueError:
            self.consoleTextEdit.append("❌ Invalid number format. Please enter a valid number:")

    def apply_deltas_and_update(self):
        """Применяет дельты к орбитальным параметрам и обновляет UI"""
        from tle_loader import parse_tle_line, update_number, build_tle_line

        try:
            line2 = self.sat_data['line2']
            deltas = self.sat_data['deltas']

            # Применяем дельты (индексы из sgp4_test.py)
            #вторая строка третье значение- наклонение орбиты line_2[2]
            #вторая строка четвертое значение- долгота восходящего угла line_2[3]
            #вторая строка пятое значение- эксцентриситет line_2[4]
            #вторая строка шестое значение- аргумент перигея line_2[5]
            #вторая строка седьмое значение- средняя аномалия line_2[6]
            #вторая строка восьмое значение- среднее движение line_2[7]

            tokens, numbers_list = parse_tle_line(line2)

            update_number(numbers_list[2], numbers_list[2][0] + deltas['inclination'])
            update_number(numbers_list[3], numbers_list[3][0] + deltas['raan'])
            update_number(numbers_list[4], numbers_list[4][0] + deltas['eccentricity'])
            update_number(numbers_list[5], numbers_list[5][0] + deltas['argument_perigee'])
            update_number(numbers_list[7], numbers_list[7][0] + deltas['mean_motion'])

            updated_line2 = build_tle_line(tokens)
            self.sat_data['line2'] = updated_line2

            self.consoleTextEdit.append("\n✔ Orbit parameters updated:")
            self.consoleTextEdit.append(f"   {self.sat_data['line0']}")
            self.consoleTextEdit.append(f"   {self.sat_data['line1']}")
            self.consoleTextEdit.append(f"   {updated_line2}\n")

            # Обновляем телеметрию с новыми значениями
            fields = parse_tle_fields(self.sat_data['line1'], updated_line2)

            self.inclination_value.setText(fields['inclination'])
            self.ascendation_value.setText(fields['raan'])
            self.eccentricity_value.setText(fields['eccentricity'])
            self.perrige_value.setText(fields['argument_perigee'])
            self.mean_motion_value.setText(fields['mean_motion'])

            self.consoleTextEdit.append("✔ Telemetry updated\n")

        except Exception as e:
            self.consoleTextEdit.append(f"❌ Error applying deltas: {e}\n")

    def send_command_pack(self):
        if not self.command_buffer:
            self.consoleTextEdit.append("[INFO] No commands to send\n")
            return

        self.consoleTextEdit.append("[PACK] Sending command pack...")
        for cmd in self.command_buffer:
            self.consoleTextEdit.append(f"[SENT] {cmd}")
        self.consoleTextEdit.append("[PACK] Done\n")
        self.command_buffer.clear()

    def run_simulation(self, duration_hours, step_minutes):
        if not self.sat_data:
            self.consoleTextEdit.append("❌ No satellite loaded. Use: sat <number>\n")
            return
        if self.loading_stage is not None:
            self.consoleTextEdit.append("❌ Finish DELTA input first\n")
            return

        self.consoleTextEdit.append(
            f"[SIM] Running simulation: {duration_hours}h, step {step_minutes} min"
        )

        start_time = datetime.now(timezone.utc)

        results = simulate(
            self.sat_data['line1'],
            self.sat_data['line2'],
            start_time,
            duration_hours=duration_hours,
            step_minutes=step_minutes
        )

        self.consoleTextEdit.append(
            f"✔ Simulation finished. Points: {len(results)}\n"
        )


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    window = MainWindow()
    window.show()

    with loop:
        loop.run_forever()
