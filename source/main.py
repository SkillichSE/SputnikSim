#  ███████╗██████╗ ██╗   ██╗████████╗███╗   ██╗██╗██╗  ██╗███████╗██╗███╗   ███╗
#  ██╔════╝██╔══██╗██║   ██║╚══██╔══╝████╗  ██║██║██║ ██╔╝██╔════╝██║████╗ ████║
#  ███████╗██████╔╝██║   ██║   ██║   ██╔██╗ ██║██║█████╔╝ ███████╗██║██╔████╔██║
#  ╚════██║██╔═══╝ ██║   ██║   ██║   ██║╚██╗██║██║██╔═██╗ ╚════██║██║██║╚██╔╝██║
#  ███████║██║     ╚██████╔╝   ██║   ██║ ╚████║██║██║  ██╗███████║██║██║ ╚═╝ ██║
#  ╚══════╝╚═╝      ╚═════╝    ╚═╝   ╚═╝  ╚═══╝╚═╝╚═╝  ╚═╝╚══════╝╚═╝╚═╝     ╚═╝

import sys
import os
import re
import asyncio
import json
from datetime import datetime, timezone
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import QLabel
from qasync import QEventLoop
from tle_loader import save_tle, dataset_sat, parse_tle_fields, find_satellite_by_norad_id
from sgp4_core import simulate, teme_to_ecef, ecef_to_latlon, gmst_from_jd, print_simulation_summary, EARTH_RADIUS
from sgp4.api import Satrec, jday
import numpy as np
import threading
import ctypes
import sys
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QIcon

def _orbit_label_from_alt(alt_km, inclination=None):
    """return a short human-readable orbit label"""
    if alt_km > 35786:
        return "GEO/GSO"
    if alt_km > 2000:
        return "MEO"
    if inclination is not None and inclination > 80:
        return "LEO Polar"
    if inclination is not None and 97.5 <= inclination <= 99.5:
        return "SSO"
    return "LEO"


class TLELoaderWorker(QThread):
    finished = pyqtSignal(bool, int)   # success, satellite_count
    progress = pyqtSignal(int, int)    # bytes_downloaded, total_bytes

    def run(self):
        try:
            count = save_tle(progress_callback=self.progress.emit)
            self.finished.emit(True, count)
        except Exception:
            self.finished.emit(False, 0)


class DeltaSuggestWorker(QThread):
    """Async worker: compute physics-based recommended orbit params for current TLE."""
    result_ready = pyqtSignal(object)  # dict or None

    def __init__(self, line0, line1, line2):
        super().__init__()
        self._line0 = line0
        self._line1 = line1
        self._line2 = line2

    def run(self):
        try:
            # AI scripts use "from parse_tle import ..." — need AI folder in path
            base = os.path.dirname(os.path.abspath(__file__))
            ai_dir = os.path.join(base, "AI")
            if ai_dir not in sys.path:
                sys.path.insert(0, ai_dir)

            from parse_tle import parse_tle
            from generate import compute_recommended_orbit_params

            _, tle_struct = parse_tle(self._line0, self._line1, self._line2)
            rec = compute_recommended_orbit_params(tle_struct)
            self.result_ready.emit(rec)
        except Exception:
            self.result_ready.emit(None)


class AIWorker(QThread):
    result_ready = pyqtSignal(str)

    def run(self):
        import subprocess
        try:
            ai_script  = os.path.join("AI", "main.py")
            tle_file   = os.path.join("AI", "data", "test.txt")
            model_best  = os.path.join("AI", "model", "tle_model_best.pth")
            model_final = os.path.join("AI", "model", "tle_model.pth")

            if not os.path.exists(model_best) and not os.path.exists(model_final):
                self.result_ready.emit(
                    "[AI] Model not found.\n\n"
                    "Please train the model first:\n"
                    "  cd AI && python train_model.py\n\n"
                    "The model will be saved to AI/model/tle_model_best.pth"
                )
                return

            chosen_model = model_best if os.path.exists(model_best) else model_final

            # force UTF-8
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            proc = subprocess.run(
                [sys.executable, ai_script,
                 "--file",       tle_file,
                 "--model",      chosen_model,
                 "--normalizer", os.path.join("AI", "model", "normalizer.pkl")],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=120,
                env=env,
            )

            output = proc.stdout.strip()
            stderr = proc.stderr.strip()

            if not output and stderr:
                stderr_clean = re.sub(r'[^\x00-\x7F\u0400-\u04FF\s]', '', stderr)
                output = f"[AI ERROR]\n{stderr_clean}"
            elif stderr:
                stderr_clean = re.sub(r'[^\x00-\x7F\u0400-\u04FF\s]', '', stderr)
                output += f"\n\n[WARNINGS]\n{stderr_clean}"

            self.result_ready.emit(output if output else "[AI] No output received.")

        except subprocess.TimeoutExpired:
            self.result_ready.emit("[AI] Analysis timed out (>120s).")
        except FileNotFoundError:
            self.result_ready.emit("[AI] AI/main.py not found. Check project structure.")
        except Exception as e:
            self.result_ready.emit(f"[AI] Error: {e}")


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("main.ui", self)
        self.setWindowIcon(QIcon("App.ico"))
        self.Answ.setReadOnly(True)
        self.consoleTextEdit.setReadOnly(True)
        self.command_buffer = []
        self.sendCommandButton.clicked.connect(self.add_command_to_console)
        self.commandLineEdit.returnPressed.connect(self.add_command_to_console)
        self.soundToggleButton.toggled.connect(self._on_sound_toggle)
        self.clearConsoleButton.clicked.connect(self._clear_console)
        self.resetSatButton.clicked.connect(self._reset_satellite)
        self.reloadTleButton.clicked.connect(self._reload_tle)

        self.current_sat_number = None
        self.loading_stage      = None
        self.sat_data           = {}
        self._ai_worker         = None

        self._live_timer = QTimer()
        self._live_timer.setInterval(30000)
        self._live_timer.timeout.connect(self._update_live_position)

        self._session_log = open("session.log", "a", encoding="utf-8")
        self._log(f"=== SESSION START {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

        self.satPoint = QLabel(self.map_pic)
        self.satPoint.setFixedSize(15, 15)
        self.satPoint.setStyleSheet("background-color: red; border-radius: 7px;")
        self.satPoint.hide()

        self.consoleTextEdit.append("[SYSTEM] Console ready")
        self.consoleTextEdit.append("  Type 'help' for available commands\n")
        self.consoleTextEdit.append("[TLE] Connecting to Celestrak...")
        # progress line
        self._tle_progress_anchor = None

        self._tle_loader = TLELoaderWorker()
        self._tle_loader.progress.connect(self._on_tle_progress)
        self._tle_loader.finished.connect(self._on_tle_loaded)
        self._tle_loader.start()

        self._delta_suggest_worker = None
        self._delta_suggestions_data = None   # last recommendations from DeltaSuggestWorker
        self._ai_general_text = ""            # last text from general AI analysis
        self._vega_mode = 'general'           # 'general' | 'delta'

        # create toggle button and float it over the AI_helper GroupBox title area
        from PyQt5.QtWidgets import QPushButton
        self.toggleModeButton = QPushButton("⇄ DELTA VALUES", self.AI_helper)
        self.toggleModeButton.setCheckable(True)
        self.toggleModeButton.setFixedSize(130, 22)
        self.toggleModeButton.setToolTip("Switch between general analysis and recommended delta values")
        self.toggleModeButton.setStyleSheet("""
            QPushButton {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 8pt;
                font-weight: bold;
                letter-spacing: 1px;
                padding: 2px 8px;
                border: 1px solid #1e7a3a;
                border-radius: 2px;
                color: #00ff88;
                background-color: transparent;
            }
            QPushButton:hover {
                background-color: #00ff88;
                color: #080c10;
            }
            QPushButton:checked {
                background-color: #0a3020;
                border-color: #00ff88;
                color: #00ff88;
            }
        """)
        # position button on the right side of the GroupBox title strip
        self.toggleModeButton.move(self.AI_helper.width() - 140, 2)
        self.toggleModeButton.raise_()
        self.toggleModeButton.show()
        self.toggleModeButton.clicked.connect(self._on_toggle_mode)

        # reposition button when window is resized
        self.AI_helper.resizeEvent = self._ai_helper_resize

    def _tle_bar(self, pct, downloaded, total):
        """build a progress string"""
        W = 30
        filled = int(W * pct / 100)
        bar = "█" * filled + "░" * (W - filled)
        if total > 0:
            return f"[TLE] [{bar}] {pct:3d}%  {downloaded // 1024} / {total // 1024} KB"
        return f"[TLE] [{bar}]  {downloaded // 1024} KB"

    def _tle_overwrite_line(self, text):
        from PyQt5.QtGui import QTextCursor
        doc = self.consoleTextEdit.document()
        if self._tle_progress_anchor is None:
            # first call
            self.consoleTextEdit.append(text)
            self._tle_progress_anchor = doc.blockCount() - 1
        else:
            block = doc.findBlockByNumber(self._tle_progress_anchor)
            if block.isValid():
                cursor = QTextCursor(block)
                cursor.movePosition(QTextCursor.StartOfBlock)
                cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
                cursor.insertText(text)
        # immediate visual update
        self.consoleTextEdit.repaint()

    def _on_tle_progress(self, downloaded, total):
        pct = min(int(downloaded / total * 100), 99) if total > 0 else 0
        self._tle_overwrite_line(self._tle_bar(pct, downloaded, total))

    def _on_tle_loaded(self, success, count):
        W = 30
        if success:
            self._tle_overwrite_line(f"[TLE] [{'█' * W}] 100%  OK — {count} satellites")
            self.consoleTextEdit.append("")
        else:
            self._tle_overwrite_line(f"[TLE] [{'░' * W}] FAILED — check connection")
            self.consoleTextEdit.append("")
        self._tle_progress_anchor = None

    def _log(self, text):
        try:
            ts = datetime.now().strftime('%H:%M:%S')
            self._session_log.write(f"[{ts}] {text}\n")
            self._session_log.flush()
        except Exception:
            pass

    def closeEvent(self, event):
        self._log(f"=== SESSION END {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        self._session_log.close()
        super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == 16777274:   # F11
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
        else:
            super().keyPressEvent(event)


    def _clear_console(self):
        self.consoleTextEdit.clear()
        self.consoleTextEdit.append("[SYSTEM] Console cleared")
        self.consoleTextEdit.append("  Type 'help' for available commands\n")

    def _reset_satellite(self):
        if not self.sat_data:
            self.consoleTextEdit.append("[ERROR] No satellite loaded\n")
            return
        norad_id = self.sat_data.get('norad_id')
        if norad_id is None:
            self.consoleTextEdit.append("[ERROR] Cannot reset — satellite loaded by index, use sat <NORAD_ID>\n")
            return
        self.consoleTextEdit.append(f"[RESET] Reloading NORAD {norad_id} with original parameters...\n")
        self.load_satellite_by_norad_id(norad_id)

    def _reload_tle(self):
        if hasattr(self, '_tle_loader') and self._tle_loader.isRunning():
            self.consoleTextEdit.append("[TLE] Already loading, please wait...\n")
            return
        self.consoleTextEdit.append("[TLE] Connecting to Celestrak...")
        self._tle_progress_anchor = None
        self._tle_loader = TLELoaderWorker()
        self._tle_loader.progress.connect(self._on_tle_progress)
        self._tle_loader.finished.connect(self._on_tle_loaded)
        self._tle_loader.start()

    def _on_sound_toggle(self, checked):
        self.soundToggleButton.setText("🔊" if checked else "🔇")

    def _play_sound(self, sound):
        """play beep (skipped if sound disabled)"""
        if not self.soundToggleButton.isChecked():
            return
        sounds = {
            'ok':      [(880, 80), (1100, 120)],
            'error':   [(300, 150), (250, 200)],
            'live_on': [(600, 80), (800, 80), (1000, 100)],
            'live_off':[(1000, 80), (700, 100)],
            'tick':    [(1200, 50)],
            'ai':      [(700, 60), (700, 50), (900, 80)],
            'sim':     [(440, 80), (660, 80), (880, 80), (1100, 120)],
            'start':   [(440, 500)],
        }
        sequence = sounds.get(sound, [])

        def _beep():
            try:
                import platform
                if platform.system() == 'Windows':
                    import winsound
                    for freq, dur in sequence:
                        winsound.Beep(freq, dur)
            except Exception:
                pass

        # run sound playback in a tiny background thread so the UI stays responsive
        if sequence:
            threading.Thread(target=_beep, daemon=True).start()

    def _write_tle_for_ai(self, line0, line1, line2):
        try:
            ai_data_dir = os.path.join("AI", "data")
            os.makedirs(ai_data_dir, exist_ok=True)
            file_path = os.path.join(ai_data_dir, "test.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"{line0}\n{line1}\n{line2}\n")
            self.consoleTextEdit.append(f"[AI] TLE saved -> {file_path}")
        except Exception as e:
            self.consoleTextEdit.append(f"[AI] Failed to write TLE file: {e}")

    def _run_ai_analysis(self):
        self._ai_general_text = ""
        if self._vega_mode == 'general':
            self.Answ.clear()
            self.Answ.append("<b>[AI]</b> Analysing satellite... please wait.")
        self._ai_worker = AIWorker()
        self._ai_worker.result_ready.connect(self._on_ai_result)
        self._ai_worker.start()

    def _on_ai_result(self, text):
        self._ai_general_text = text
        if self._vega_mode == 'general':
            self.Answ.clear()
            self.Answ.append(text)
        self._play_sound('ai')

    def _update_live_position(self):
        if not self.sat_data:
            return
        try:
            sat = Satrec.twoline2rv(self.sat_data['line1'], self.sat_data['line2'])
            now = datetime.now(timezone.utc)
            jd, fr = jday(now.year, now.month, now.day,
                          now.hour, now.minute, now.second)
            error, r_teme, _ = sat.sgp4(jd, fr)
            if error == 0:
                r_teme = np.array(r_teme)
                r_ecef = teme_to_ecef(r_teme, jd + fr)
                lat, lon = ecef_to_latlon(r_ecef)
                self.update_satellite_marker(lat, lon)
                self.consoleTextEdit.append(
                    f"[POS] {now.strftime('%H:%M:%S')} UTC  "
                    f"LAT {lat:+.3f}  LON {lon:+.3f}"
                )
        except Exception:
            pass

    def add_command_to_console(self):
        command = self.commandLineEdit.text().strip()
        if not command:
            return

        self.consoleTextEdit.append(f"> {command}")
        self.commandLineEdit.clear()
        self._log(f"> {command}")
        self._play_sound('tick')

        parts = command.split()

        # allow sat/help/live commands even during delta input
        if self.loading_stage is not None and (not parts or parts[0].lower() not in ('sat', 'help', 'live', 'simulate')):
            self.process_satellite_parameter(command)
            return

        if parts[0].lower() == "help":
            self.consoleTextEdit.append(
                "\n══════════════════════════════════════════════\n"
                "  ДОСТУПНЫЕ КОМАНДЫ\n"
                "══════════════════════════════════════════════\n"
                "  sat <NORAD_ID>        — загрузить спутник по номеру\n"
                "  sat <имя>             — поиск спутника по имени\n"
                "  delta                 — изменить параметры орбиты\n"
                "  simulate <ч> <шаг>    — симуляция трека (часы, мин)\n"
                "  live                  — вкл/выкл отслеживание (30 с)\n"
                "  help                  — эта справка\n"
                "  F11                   — полноэкранный режим\n"
                "\n"
                "  ПРИМЕРЫ:\n"
                "  sat 25544             — МКС (LEO, ~408 км)\n"
                "  sat 28654             — GPS IIR-1 (MEO, ~20 000 км)\n"
                "  sat 37820             — INTELSAT 33E (GEO, ~35 786 км)\n"
                "  sat STARLINK          — поиск Starlink-спутников\n"
                "  simulate 24 1         — симуляция 24 ч, шаг 1 мин\n"
                "  simulate 2 0.5        — симуляция 2 ч, шаг 30 с\n"
                "══════════════════════════════════════════════\n"
            )
            return

        if parts[0].lower() == "simulate":
            try:
                duration_hours = float(parts[1])
                step_minutes   = float(parts[2])
            except (IndexError, ValueError):
                self.consoleTextEdit.append("Usage: simulate <hours> <step_minutes>\n")
                return
            self.run_simulation(duration_hours, step_minutes)
            return

        if parts[0].lower() == "sat":
            if len(parts) < 2:
                self.consoleTextEdit.append(
                    "Usage:\n"
                    "  sat <NORAD_ID>  - load by NORAD ID (e.g., sat 25544)\n"
                    "  sat <name>      - search by name (e.g., sat STARLINK)\n"
                )
                return
            if parts[1].isdigit():
                self.load_satellite_by_norad_id(int(parts[1]))
            else:
                self.search_satellite_by_name(" ".join(parts[1:]))
            return

        if parts[0].lower() == "delta":
            if not self.sat_data:
                self.consoleTextEdit.append("[ERROR] No satellite loaded. Use: sat <NORAD_ID>\n")
                return
            self.loading_stage = 'inclination'

            # auto-switch Answ to delta recommendations mode
            if not self.toggleModeButton.isChecked():
                self.toggleModeButton.setChecked(True)
                self._on_toggle_mode(True)

            self.consoleTextEdit.append("[INPUT] Enter new orbit parameters:")
            self.consoleTextEdit.append("Inclination (deg):")
            return

        if parts[0].lower() == "live":
            if not self.sat_data:
                self.consoleTextEdit.append("[ERROR] No satellite loaded. Use: sat <NORAD_ID>\n")
                return
            if self._live_timer.isActive():
                self._live_timer.stop()
                self.consoleTextEdit.append("[STOP] Live tracking stopped\n")
                self._play_sound('live_off')
            else:
                self._live_timer.start()
                self._update_live_position()
                orbit = self.sat_data.get('orbit_label', 'unknown')
                self.consoleTextEdit.append(
                    f"[LIVE] Live tracking started (updates every 30s) — {orbit}\n"
                )
                self._play_sound('live_on')
            return

        self.command_buffer.append(command)

    def search_satellite_by_name(self, query):
        try:
            with open("data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            self.consoleTextEdit.append(f"[ERROR] Cannot read TLE database: {e}\n")
            return

        query_upper = query.upper()
        matches = []
        for i in range(0, len(data) - 2, 3):
            name  = data[i].strip()
            line1 = data[i + 1].strip()
            if query_upper in name.upper():
                norad_id = line1[2:7].strip()
                matches.append((norad_id, name))

        if not matches:
            self.consoleTextEdit.append(f"[ERROR] No satellites found matching '{query}'\n")
            return

        if len(matches) == 1:
            self.consoleTextEdit.append(f"[OK] Found: {matches[0][1]}")
            self.load_satellite_by_norad_id(int(matches[0][0]))
        else:
            self.consoleTextEdit.append(f"Found {len(matches)} satellites matching '{query}':")
            for norad_id, name in matches[:10]:
                self.consoleTextEdit.append(f"  NORAD {norad_id} - {name}")
            if len(matches) > 10:
                self.consoleTextEdit.append(f"  ... and {len(matches) - 10} more. Refine your search.\n")
            else:
                self.consoleTextEdit.append("  Use: sat <NORAD_ID> to load one of the above\n")

    def load_satellite_by_norad_id(self, norad_id):
        """load a satellite by NORAD"""
        try:
            result = find_satellite_by_norad_id(norad_id)

            if result is None:
                self.consoleTextEdit.append(
                    f"[ERROR] Satellite NORAD {norad_id} not found\n"
                    f"        Check https://celestrak.org for valid IDs\n"
                )
                self._play_sound('error')
                return

            line0, line1, line2 = result

            self.consoleTextEdit.append(f"[OK] Satellite found: {line0}")
            self._play_sound('ok')
            self.consoleTextEdit.append(f"   {line1}")
            self.consoleTextEdit.append(f"   {line2}\n")

            self.sat_data = {
                'norad_id': norad_id,
                'line0':    line0,
                'line1':    line1,
                'line2':    line2,
                'deltas':   {},
            }

            fields = parse_tle_fields(line1, line2)
            self._populate_ui_fields(line0, fields)

            self._write_tle_for_ai(line0, line1, line2)
            self._run_ai_analysis()
            self._delta_suggestions_data = None  # reset stale data for new satellite
            self._start_delta_suggestions()
            try:
                _sat = Satrec.twoline2rv(line1, line2)
                _now = datetime.now(timezone.utc)
                _jd, _fr = jday(_now.year, _now.month, _now.day,
                                _now.hour, _now.minute, _now.second)
                _err, _r, _ = _sat.sgp4(_jd, _fr)
                if _err == 0:
                    _r    = np.array(_r)
                    _ecef = teme_to_ecef(_r, _jd + _fr)
                    _lat, _lon = ecef_to_latlon(_ecef)
                    _alt  = np.linalg.norm(_r) - EARTH_RADIUS
                    _incl = float(fields.get('inclination', 0))
                    orbit = _orbit_label_from_alt(_alt, _incl)

                    self.sat_data['altitude_km']  = _alt
                    self.sat_data['orbit_label']  = orbit
                    self.update_satellite_marker(_lat, _lon)

                    self.consoleTextEdit.append(
                        f"[POS] {_now.strftime('%H:%M:%S')} UTC  "
                        f"LAT {_lat:+.3f}  LON {_lon:+.3f}  ALT {_alt:.0f} km  [{orbit}]"
                    )

                    # orbit type info
                    if _alt > 35000:
                        self.consoleTextEdit.append(
                            "[ORBIT] GEO/GSO — почти стационарная орбита. "
                            "Смещение менее 1° в сутки.\n"
                        )
                    elif _alt > 2000:
                        self.consoleTextEdit.append(
                            "[ORBIT] MEO — средняя орбита. "
                            "Полный оборот ~12 ч.\n"
                        )
                    else:
                        self.consoleTextEdit.append(
                            "[ORBIT] LEO — быстрое движение, ~90 мин/оборот. "
                            "Используйте 'simulate' для просмотра трека.\n"
                        )
            except Exception:
                pass

            self.consoleTextEdit.append(
                "─────────────────────────────────────────────\n"
                "  delta               — изменить параметры орбиты\n"
                "  live                — включить/выключить отслеживание (30 с)\n"
                "  simulate <h> <step> — симуляция трека (ч, мин)\n"
                "─────────────────────────────────────────────\n"
            )
            self._log(f"Loaded satellite: {line0} (NORAD {norad_id})")

        except Exception as e:
            self.consoleTextEdit.append(f"[ERROR] Failed to load satellite: {e}\n")
            self._play_sound('error')

    def _populate_ui_fields(self, line0, fields):
        """fill all telemetry labels in the UI"""
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
        self.pressure_value.setText(fields['bstar'])
        self.mean_motion_value.setText(fields['mean_motion'])
        self.ephemeris_value.setText(fields['ephemeris_type'])
        self.element_value.setText(fields['element_number'])
        self.rottation_value.setText(fields['revolution_number'])

    def start_satellite_loading(self, sat_number):
        try:
            number_sat = (sat_number - 1) * 3
            line0, line1, line2 = dataset_sat(number_sat)

            self.consoleTextEdit.append(f"[OK] Satellite found: {line0}")
            self._play_sound('ok')
            self.consoleTextEdit.append(f"   {line1}")
            self.consoleTextEdit.append(f"   {line2}\n")

            self.sat_data = {
                'sat_number': sat_number,
                'line0': line0,
                'line1': line1,
                'line2': line2,
                'deltas': {},
            }

            fields = parse_tle_fields(line1, line2)
            self._populate_ui_fields(line0, fields)

            self._write_tle_for_ai(line0, line1, line2)
            self._run_ai_analysis()
            self._delta_suggestions_data = None  # reset stale data for new satellite
            self._start_delta_suggestions()
            self._update_live_position()

            self.consoleTextEdit.append(
                "─────────────────────────────────────────────\n"
                "  delta               — изменить параметры орбиты\n"
                "  live                — включить/выключить отслеживание (30 с)\n"
                "  simulate <h> <step> — симуляция трека (ч, мин)\n"
                "─────────────────────────────────────────────\n"
            )
            self._log(f"Loaded satellite: {line0.strip()}")

        except Exception as e:
            self.consoleTextEdit.append(f"[ERROR] Error loading satellite: {e}")
            self._play_sound('error')
            self.loading_stage = None

    def process_satellite_parameter(self, value):
        try:
            param_value = float(value)

            if self.loading_stage == 'inclination':
                self.sat_data['deltas']['inclination'] = param_value
                self.loading_stage = 'raan'
                self.consoleTextEdit.append("RAAN (deg):")

            elif self.loading_stage == 'raan':
                self.sat_data['deltas']['raan'] = param_value
                self.loading_stage = 'eccentricity'
                self.consoleTextEdit.append("Eccentricity (0.x or 000xxxx):")

            elif self.loading_stage == 'eccentricity':
                self.sat_data['deltas']['eccentricity'] = param_value
                self.loading_stage = 'argument_perigee'
                self.consoleTextEdit.append("Argument of Perigee (deg):")

            elif self.loading_stage == 'argument_perigee':
                self.sat_data['deltas']['argument_perigee'] = param_value
                self.loading_stage = 'mean_motion'
                self.consoleTextEdit.append("Mean Motion (rev/day):")

            elif self.loading_stage == 'mean_motion':
                self.sat_data['deltas']['mean_motion'] = param_value
                self.loading_stage = None
                self.consoleTextEdit.append("[OK] Parameter input completed\n")
                self._play_sound('ok')
                self.apply_deltas_and_update()

        except ValueError:
            self.consoleTextEdit.append("[ERROR] Invalid number format. Please enter a valid number:")
            self._play_sound('error')

    def apply_deltas_and_update(self):
        from tle_loader import parse_tle_line, update_number, build_tle_line, fix_checksum
        try:
            line2  = self.sat_data['line2']
            deltas = self.sat_data['deltas']

            tokens, numbers_list = parse_tle_line(line2)

            # use absolute values provided by the user instead of deltas
            if 'inclination' in deltas:
                update_number(numbers_list[2], deltas['inclination'])
            if 'raan' in deltas:
                update_number(numbers_list[3], deltas['raan'])
            if 'eccentricity' in deltas:
                # TLE stores eccentricity as 7‑digit fractional part without decimal point.
                # Accept either raw TLE-style value (e.g. 0008356 / 8356)
                # or physical eccentricity (e.g. 0.0008356).
                ecc_old_text = numbers_list[4][1]
                width = len(ecc_old_text)
                ecc_val = deltas['eccentricity']
                if abs(ecc_val) < 1.0:
                    scaled = int(round(ecc_val * (10 ** width)))
                else:
                    scaled = int(round(ecc_val))
                update_number(numbers_list[4], scaled)
            if 'argument_perigee' in deltas:
                update_number(numbers_list[5], deltas['argument_perigee'])
            if 'mean_motion' in deltas:
                update_number(numbers_list[7], deltas['mean_motion'])

            updated_line2 = fix_checksum(build_tle_line(tokens))
            self.sat_data['line2'] = updated_line2

            # try to persist updated TLE back into data.json so that
            # subsequent loads of this satellite see the new line 2
            try:
                db_path = os.path.join("data.json")
                if os.path.exists(db_path):
                    with open(db_path, "r", encoding="utf-8") as f:
                        db_data = json.load(f)
                    line0 = self.sat_data.get('line0') or self.sat_data.get('name', '')
                    line1 = self.sat_data.get('line1', '')
                    # find matching triplet and replace only line 2
                    for i in range(0, len(db_data) - 2, 3):
                        if db_data[i] == line0 and db_data[i + 1] == line1 and db_data[i + 2] == line2:
                            db_data[i + 2] = updated_line2
                            with open(db_path, "w", encoding="utf-8") as f:
                                json.dump(db_data, f)
                            break
            except Exception:
                # if persistence fails, keep working with in‑memory TLE
                pass

            self.consoleTextEdit.append("\n[OK] Orbit parameters updated:")
            self.consoleTextEdit.append(f"   {self.sat_data['line0']}")
            self.consoleTextEdit.append(f"   {self.sat_data['line1']}")
            self.consoleTextEdit.append(f"   {updated_line2}\n")

            fields = parse_tle_fields(self.sat_data['line1'], updated_line2)

            self.inclination_value.setText(fields['inclination'])
            self.ascendation_value.setText(fields['raan'])
            self.eccentricity_value.setText(fields['eccentricity'])
            self.perrige_value.setText(fields['argument_perigee'])
            self.mean_motion_value.setText(fields['mean_motion'])

            self.consoleTextEdit.append("[OK] Telemetry updated\n")

            self._write_tle_for_ai(self.sat_data['line0'], self.sat_data['line1'], updated_line2)
            self._run_ai_analysis()
            self._update_live_position()

            self._log(f"Delta applied: {deltas}")

        except Exception as e:
            self.consoleTextEdit.append(f"[ERROR] Error applying deltas: {e}\n")
            self._play_sound('error')

    def run_simulation(self, duration_hours, step_minutes):
        if not self.sat_data:
            self.consoleTextEdit.append("[ERROR] No satellite loaded. Use: sat <NORAD_ID>\n")
            return
        if self.loading_stage is not None:
            self.consoleTextEdit.append("[ERROR] Finish parameter input first\n")
            return

        orbit = self.sat_data.get('orbit_label', '')
        self.consoleTextEdit.append(
            f"[SIM] Running simulation: {duration_hours}h, step {step_minutes} min"
            + (f"  [{orbit}]" if orbit else "") + "\n"
        )

        start_time = datetime.now(timezone.utc)
        results, summary = simulate(
            self.sat_data['line1'],
            self.sat_data['line2'],
            start_time,
            duration_hours=duration_hours,
            step_minutes=step_minutes,
        )

        summary_text = print_simulation_summary(summary, console_output=False)
        for line in summary_text.split('\n'):
            self.consoleTextEdit.append(line)

        self.consoleTextEdit.append("")

        if results and 'lat' in results[-1]:
            last = results[-1]
            self.update_satellite_marker(last['lat'], last['lon'])
            self.consoleTextEdit.append(
                f"[OK] Final position: LAT {last['lat']:.3f}  "
                f"LON {last['lon']:.3f}  ALT {last['altitude_km']:.2f} km\n"
            )

        self._play_sound('sim')
        self._log(f"Simulation: {duration_hours}h step={step_minutes}min points={len(results)}")

    def _start_delta_suggestions(self):
        """Start async AI recommendations for delta; display in Answ when done."""
        line0 = self.sat_data.get('line0')
        line1 = self.sat_data.get('line1')
        line2 = self.sat_data.get('line2')
        if not line0 or not line1 or not line2:
            if self._vega_mode == 'delta':
                self.Answ.setPlainText("[AI] Рекомендации недоступны: нет актуального TLE.")
            return

        if self._vega_mode == 'delta':
            self.Answ.setPlainText("[AI] Загрузка рекомендаций для коррекции орбиты…")

        self._delta_suggest_worker = DeltaSuggestWorker(line0, line1, line2)
        self._delta_suggest_worker.result_ready.connect(self._on_delta_suggestions_ready)
        self._delta_suggest_worker.start()

    def _on_delta_suggestions_ready(self, data):
        """Display AI recommendations in Answ."""
        self._delta_suggestions_data = data
        if self._vega_mode == 'delta':
            self._show_delta_suggestions()

    def _show_delta_suggestions(self):
        """Render saved delta suggestions into Answ."""
        data = self._delta_suggestions_data
        if data is None:
            self.Answ.setPlainText(
                "[AI] Не удалось получить рекомендации — будут использованы только ваши значения."
            )
            return
        orbit_type = data.get("orbit_type", "")
        ecc = max(0.0, float(data.get("eccentricity", 0)))
        self.Answ.setPlainText(
            f"[AI] Рекомендуемые параметры орбиты ({orbit_type}):\n\n"
            f"Inclination:      {data.get('inclination', 0):.4f}°\n"
            f"RAAN:             {data.get('raan', 0):.4f}°\n"
            f"Eccentricity:     {ecc:.6f}\n"
            f"Arg. Perigee:     {data.get('argument_perigee', 0):.4f}°\n"
            f"Mean Motion:      {data.get('mean_motion', 0):.8f} rev/day\n\n"
        )

    def _ai_helper_resize(self, event):
        """Keep the toggle button pinned to the right of the GroupBox title on resize."""
        self.toggleModeButton.move(self.AI_helper.width() - 140, 2)
        type(self.AI_helper).resizeEvent(self.AI_helper, event)

    def _on_toggle_mode(self, checked):
        """Switch Answ between general AI analysis and delta recommendations."""
        if checked:
            self._vega_mode = 'delta'
            self.toggleModeButton.setText("⇄ GENERAL")
            if self._delta_suggestions_data is not None:
                self._show_delta_suggestions()
            else:
                self.Answ.setPlainText(
                    "[AI] Рекомендуемые значения delta недоступны.\n"
                    "Выполните команду 'delta' для их загрузки."
                )
        else:
            self._vega_mode = 'general'
            self.toggleModeButton.setText("⇄ DELTA VALUES")
            if self._ai_general_text:
                self.Answ.clear()
                self.Answ.append(self._ai_general_text)
            else:
                self.Answ.setPlainText("[AI] Общий анализ недоступен.")

    def update_satellite_marker(self, lat, lon):
        width  = self.map_pic.width()
        height = self.map_pic.height()

        # equirectangular projection
        x = (lon + 180) / 360 * width
        y = (90 - lat) / 180 * height

        self.satPoint.move(int(x) - 5, int(y) - 5)
        self.satPoint.show()


if __name__ == "__main__":
    app  = QtWidgets.QApplication(sys.argv)
    import platform
    if platform.system() == 'Windows':
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Sput.TLE.App")
    app.setWindowIcon(QIcon(os.path.join(os.path.dirname(os.path.abspath(__file__)), "App.ico")))

    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    window = MainWindow()
    window.show()
    window._play_sound('start')

    with loop:
        loop.run_forever()