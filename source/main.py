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
from PyQt5 import QtWidgets, QtGui, QtCore, uic
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt, QPointF, QRect
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
from PyQt5.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QFont, QRadialGradient, QBrush

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



class SimulateWorker(QThread):
    """Run orbit simulation in background thread."""
    result_ready = pyqtSignal(object, object)  # results, summary
    error        = pyqtSignal(str)

    def __init__(self, line1, line2, start_time, duration_hours, step_minutes):
        super().__init__()
        self._line1           = line1
        self._line2           = line2
        self._start_time      = start_time
        self._duration_hours  = duration_hours
        self._step_minutes    = step_minutes

    def run(self):
        try:
            results, summary = simulate(
                self._line1, self._line2, self._start_time,
                duration_hours=self._duration_hours,
                step_minutes=self._step_minutes,
            )
            self.result_ready.emit(results, summary)
        except Exception as e:
            self.error.emit(str(e))


class SplashScreen(QtWidgets.QSplashScreen):
    """Animated splash screen with orbiting satellite."""

    def __init__(self):
        # create blank pixmap — we draw everything ourselves
        px = QPixmap(520, 340)
        px.fill(QtGui.QColor(0, 0, 0, 0))
        super().__init__(px, Qt.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.FramelessWindowHint)

        self._angle = 0.0
        self._dots  = 0

        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

        self._status = "Initializing..."

    def set_status(self, text):
        self._status = text
        self._tick()

    def _tick(self):
        self._angle = (self._angle + 2.5) % 360
        self._redraw()

    def _redraw(self):
        import math
        W, H = 520, 340
        px = QPixmap(W, H)
        px.fill(QtGui.QColor(8, 12, 24))

        p = QPainter(px)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        # background stars (static seed)
        import random
        rng = random.Random(42)
        for _ in range(80):
            sx = rng.randint(0, W)
            sy = rng.randint(0, H)
            sr = rng.uniform(0.5, 1.5)
            alpha = rng.randint(80, 220)
            p.setBrush(QtGui.QColor(255, 255, 255, alpha))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QtCore.QPointF(sx, sy), sr, sr)

        cx, cy = W // 2, H // 2 - 20
        R_logo = 68          # logo circle radius (same as old R_earth)
        R_orb_x, R_orb_y = 130, 50

        rad = math.radians(self._angle)
        sat_x = cx + R_orb_x * math.cos(rad)
        sat_y = cy + R_orb_y * math.sin(rad)
        behind = math.sin(rad) < 0

        # ── back half orbit dashes ──────────────────────────
        pen_orb_back = QPen(QtGui.QColor(80, 160, 255, 35), 1, Qt.DashLine)
        p.setPen(pen_orb_back)
        p.setBrush(Qt.NoBrush)
        p.drawArc(
            int(cx - R_orb_x), int(cy - R_orb_y),
            int(R_orb_x * 2), int(R_orb_y * 2),
            0, 180 * 16
        )

        # ── satellite & trail when BEHIND logo ──────────────
        if behind:
            for i in range(1, 12):
                a2 = math.radians(self._angle - i * 4)
                if math.sin(a2) >= 0:
                    break
                tx = cx + R_orb_x * math.cos(a2)
                ty = cy + R_orb_y * math.sin(a2)
                alpha = int(100 * (1 - i / 12))
                p.setBrush(QtGui.QColor(0, 200, 255, alpha))
                p.setPen(Qt.NoPen)
                p.drawEllipse(QtCore.QPointF(tx, ty), 1.5, 1.5)
            p.save()
            p.translate(sat_x, sat_y)
            p.rotate(self._angle)
            p.setOpacity(0.25)
            p.setBrush(QtGui.QColor(200, 220, 255))
            p.setPen(QPen(QtGui.QColor(100, 180, 255), 1))
            p.drawRect(-5, -3, 10, 6)
            p.setBrush(QtGui.QColor(0, 120, 255, 180))
            p.drawRect(-14, -1, 8, 2)
            p.drawRect(6,   -1, 8, 2)
            p.setOpacity(1.0)
            p.restore()

        # ── logo (drawn on top of back-half satellite) ───────
        # outer glow ring
        glow = QtGui.QRadialGradient(cx, cy, R_logo * 1.45)
        glow.setColorAt(0.70, QtGui.QColor(40, 120, 255, 0))
        glow.setColorAt(1.00, QtGui.QColor(40, 120, 255, 60))
        p.setBrush(QtGui.QBrush(glow))
        p.setPen(Qt.NoPen)
        p.drawEllipse(QtCore.QPointF(cx, cy), R_logo * 1.45, R_logo * 1.45)

        # try to load ico.jpg; fall back to a dark circle if missing
        logo_src = QPixmap("ico.png")
        if not logo_src.isNull():
            # clip to circle
            diameter = int(R_logo * 2)
            logo_round = QPixmap(diameter, diameter)
            logo_round.fill(Qt.transparent)
            lp = QPainter(logo_round)
            lp.setRenderHint(QPainter.Antialiasing)
            lp.setRenderHint(QPainter.SmoothPixmapTransform)
            from PyQt5.QtGui import QPainterPath
            path = QPainterPath()
            path.addEllipse(0, 0, diameter, diameter)
            lp.setClipPath(path)
            scaled = logo_src.scaled(diameter, diameter,
                                     Qt.KeepAspectRatioByExpanding,
                                     Qt.SmoothTransformation)
            # centre-crop
            ox = (scaled.width()  - diameter) // 2
            oy = (scaled.height() - diameter) // 2
            lp.drawPixmap(0, 0, scaled, ox, oy, diameter, diameter)
            lp.end()
            p.drawPixmap(int(cx - R_logo), int(cy - R_logo), logo_round)
        else:
            # fallback — plain dark circle
            grad = QtGui.QRadialGradient(cx - 15, cy - 15, R_logo * 1.2)
            grad.setColorAt(0.0, QtGui.QColor(30, 80, 180))
            grad.setColorAt(0.6, QtGui.QColor(20, 120, 60))
            grad.setColorAt(1.0, QtGui.QColor(10, 40, 120))
            p.setBrush(QtGui.QBrush(grad))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QtCore.QPointF(cx, cy), R_logo, R_logo)

        # thin border ring over logo
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QtGui.QColor(60, 140, 255, 100), 1.5))
        p.drawEllipse(QtCore.QPointF(cx, cy), R_logo, R_logo)

        # ── front half orbit dashes ──────────────────────────
        pen_orb_front = QPen(QtGui.QColor(80, 160, 255, 70), 1, Qt.DashLine)
        p.setPen(pen_orb_front)
        p.setBrush(Qt.NoBrush)
        p.drawArc(
            int(cx - R_orb_x), int(cy - R_orb_y),
            int(R_orb_x * 2), int(R_orb_y * 2),
            180 * 16, 180 * 16
        )

        # ── satellite & trail when IN FRONT of logo ──────────
        if not behind:
            for i in range(1, 14):
                a2 = math.radians(self._angle - i * 4)
                tx = cx + R_orb_x * math.cos(a2)
                ty = cy + R_orb_y * math.sin(a2)
                alpha = int(180 * (1 - i / 14))
                p.setBrush(QtGui.QColor(0, 200, 255, alpha))
                p.setPen(Qt.NoPen)
                r2 = 2.5 * (1 - i / 14)
                p.drawEllipse(QtCore.QPointF(tx, ty), r2, r2)
            p.save()
            p.translate(sat_x, sat_y)
            p.rotate(self._angle)
            p.setBrush(QtGui.QColor(200, 220, 255))
            p.setPen(QPen(QtGui.QColor(100, 180, 255), 1))
            p.drawRect(-5, -3, 10, 6)
            p.setBrush(QtGui.QColor(0, 120, 255, 200))
            p.drawRect(-14, -1, 8, 2)
            p.drawRect(6,   -1, 8, 2)
            p.restore()

        # ── title ────────────────────────────────────────────
        p.setPen(QtGui.QColor(255, 255, 255))
        f = QtGui.QFont("Consolas", 26, QtGui.QFont.Bold)
        p.setFont(f)
        p.drawText(QtCore.QRect(0, H - 95, W, 36), Qt.AlignCenter, "SputnikSim")

        p.setPen(QtGui.QColor(80, 160, 255))
        f2 = QtGui.QFont("Consolas", 9)
        p.setFont(f2)
        p.drawText(QtCore.QRect(0, H - 62, W, 20), Qt.AlignCenter,
                   "Satellite Orbit Analysis & Simulation System")

        # ── progress bar (smooth, no dots) ───────────────────
        bar_w, bar_h = 320, 4
        bar_x = (W - bar_w) // 2
        bar_y = H - 28
        # background track
        p.setBrush(QtGui.QColor(40, 50, 70))
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 2, 2)
        # animated fill — bouncing scanner effect
        import math as _m
        phase = (self._angle / 360.0) % 1.0
        fill_w = int(bar_w * (0.4 + 0.35 * abs(_m.sin(phase * _m.pi * 2))))
        fill_x = bar_x + int((bar_w - fill_w) * (0.5 + 0.5 * _m.sin(phase * _m.pi * 2)))
        bar_grad = QtGui.QLinearGradient(fill_x, 0, fill_x + fill_w, 0)
        bar_grad.setColorAt(0.0, QtGui.QColor(0, 120, 255, 0))
        bar_grad.setColorAt(0.3, QtGui.QColor(0, 200, 255, 255))
        bar_grad.setColorAt(0.7, QtGui.QColor(0, 200, 255, 255))
        bar_grad.setColorAt(1.0, QtGui.QColor(0, 120, 255, 0))
        p.setBrush(QtGui.QBrush(bar_grad))
        p.drawRoundedRect(fill_x, bar_y, fill_w, bar_h, 2, 2)

        # status text (no dots — static)
        p.setPen(QtGui.QColor(140, 200, 100))
        f3 = QtGui.QFont("Consolas", 8)
        p.setFont(f3)
        p.drawText(QtCore.QRect(0, H - 18, W, 16), Qt.AlignCenter, self._status)

        p.end()
        self.setPixmap(px)

    def mousePressEvent(self, event):
        pass  # ignore clicks — do not close on click

    def stop(self):
        self._timer.stop()



class MapWidget(QLabel):
    """QLabel with trajectory and satellite marker painted on top."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._trajectory = []
        self._sat_pos = None

    def set_trajectory(self, points):
        self._trajectory = points
        self.update()

    def set_satellite(self, lat, lon):
        self._sat_pos = (lat, lon)
        self.update()

    def clear_trajectory(self):
        self._trajectory = []
        self.update()

    def _to_xy(self, lat, lon):
        w, h = self.width(), self.height()
        return int((lon + 180) / 360 * w), int((90 - lat) / 180 * h)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if len(self._trajectory) >= 2:
            pen = QPen(QColor(0, 180, 255, 200))
            pen.setWidth(2)
            painter.setPen(pen)
            prev = None
            for lat, lon in self._trajectory:
                x, y = self._to_xy(lat, lon)
                if prev and abs(x - prev[0]) < self.width() * 0.5:
                    painter.drawLine(prev[0], prev[1], x, y)
                prev = (x, y)

        if self._sat_pos:
            lat, lon = self._sat_pos
            x, y = self._to_xy(lat, lon)
            painter.setPen(QPen(QColor(255, 50, 50), 2))
            painter.setBrush(QColor(255, 50, 50))
            painter.drawEllipse(x - 6, y - 6, 12, 12)

        painter.end()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("main.ui", self)
        self.setWindowIcon(QIcon("App.ico"))
        self.Answ.setReadOnly(True)
        self.consoleTextEdit.setReadOnly(True)
        self.consoleTextEdit.setOpenLinks(False)
        self.consoleTextEdit.anchorClicked.connect(
            lambda url: __import__("webbrowser").open(url.toString()))
        _plain_fmt = QtGui.QTextCharFormat()
        _plain_fmt.setAnchor(False)
        _plain_fmt.setAnchorHref("")
        _plain_fmt.setFontUnderline(False)
        _plain_fmt.setForeground(QtGui.QColor(168, 228, 255))

        _orig_append = self.consoleTextEdit.append

        def _safe_append(text, _fmt=_plain_fmt, _orig=_orig_append):
            self.consoleTextEdit.setCurrentCharFormat(_fmt)
            _orig(text)
            self.consoleTextEdit.setCurrentCharFormat(_fmt)

        self.consoleTextEdit.append = _safe_append
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
        self._sim_worker        = None
        self._sim_workers       = []

        self._live_timer = QTimer()
        self._live_timer.setInterval(30000)
        self._live_timer.timeout.connect(self._update_live_position)

        self._session_log = open("session.log", "a", encoding="utf-8")
        self._log(f"=== SESSION START {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

        # inject MapWidget in place of map_pic QLabel
        self._map_widget = MapWidget()
        _px = self.map_pic.pixmap()
        if _px:
            self._map_widget.setPixmap(_px)
        else:
            self._map_widget.setPixmap(QPixmap('map.jpg'))
        self._map_widget.setScaledContents(True)
        _layout = self.map_pic.parent().layout()
        if _layout:
            _layout.replaceWidget(self.map_pic, self._map_widget)
        self.map_pic.hide()
        self.map_pic = self._map_widget

        # progress line
        self._tle_progress_anchor = None

        self._tle_loader = TLELoaderWorker()
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

    def _on_tle_loaded(self, success, count):
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
                "\nДОСТУПНЫЕ КОМАНДЫ\n"
                "\n"
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
                "  sat 28129             — GPS BIIR-10 (MEO, ~20 200 км)\n"
                "  sat 36101             — Eutelsat 36B (GEO, ~35 786 км)\n"
                "  sat STARLINK          — поиск Starlink-спутников\n"
                "  simulate 24 1         — симуляция 24 ч, шаг 1 мин\n"
                "  simulate 2 0.5        — симуляция 2 ч, шаг 30 с\n"
                "\n"
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
        # You found an easter egg, nice code check
        if parts[0].lower() == "skis":
            from PyQt5.QtGui import QTextCursor
            cursor = self.consoleTextEdit.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.insertBlock()
            cursor.insertHtml(
                "<span style='color:#00cfff;'>[SKIS]</span>"
                "&nbsp;Oh, you found some easter egg! It's a site one of the developer:&nbsp;"
                "<a href='https://skillichse.github.io/portfolio/' style='color:#44aaff;'>"
                "https://skillichse.github.io/portfolio/</a>"
            )
            self.consoleTextEdit.setTextCursor(cursor)
            self.consoleTextEdit.append("")
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

            self.map_pic.clear_trajectory()
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
                "\n  delta               — изменить параметры орбиты\n"
                "  live                — включить/выключить отслеживание (30 с)\n"
                "  simulate <h> <step> — симуляция трека (ч, мин)\n"
                "\n"
            )
            if norad_id == 25544:
                from PyQt5.QtGui import QTextCursor, QTextCharFormat, QTextBlockFormat
                cursor = self.consoleTextEdit.textCursor()
                cursor.movePosition(QTextCursor.End)
                cursor.insertBlock()
                # insert the hyperlink
                cursor.insertHtml(
                    "<span style='color:#00cfff;'>[ISS]</span>"
                    "&nbsp;Live stream:&nbsp;"
                    "<a href='https://isslivenow.com/' style='color:#44aaff;'>"
                    "https://isslivenow.com/</a>"
                )
                # move to end and insert a clean block — reset ALL formatting
                cursor.movePosition(QTextCursor.End)
                cursor.insertBlock()
                plain_char = QTextCharFormat()
                plain_char.setAnchor(False)
                plain_char.setAnchorHref("")
                plain_char.setForeground(QtGui.QColor(168, 228, 255))  # default console colour
                plain_char.setFontUnderline(False)
                cursor.setCharFormat(plain_char)
                cursor.setBlockCharFormat(plain_char)
                cursor.insertBlock()
                cursor.setCharFormat(plain_char)
                cursor.setBlockCharFormat(plain_char)
                self.consoleTextEdit.setTextCursor(cursor)
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
                "\n  delta               — изменить параметры орбиты\n"
                "  live                — включить/выключить отслеживание (30 с)\n"
                "  simulate <h> <step> — симуляция трека (ч, мин)\n"
                "\n"
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
        if hasattr(self, '_sim_worker') and self._sim_worker and self._sim_worker.isRunning():
            self.consoleTextEdit.append("[SIM] Simulation already running, please wait...\n")
            return

        orbit = self.sat_data.get('orbit_label', '')
        self.consoleTextEdit.append(
            f"[SIM] Running simulation: {duration_hours}h, step {step_minutes} min"
            + (f"  [{orbit}]" if orbit else "") + "\n"
        )

        start_time = datetime.now(timezone.utc)
        w = SimulateWorker(
            self.sat_data['line1'], self.sat_data['line2'],
            start_time, duration_hours, step_minutes,
        )
        self._sim_worker = w
        self._sim_workers.append(w)
        w.result_ready.connect(
            lambda res, summ: self._on_simulation_done(res, summ, duration_hours, step_minutes)
        )
        w.error.connect(
            lambda e: self.consoleTextEdit.append(f"[SIM ERROR] {e}\n")
        )
        w.finished.connect(lambda _w=w: self._sim_workers.remove(_w) if _w in self._sim_workers else None)
        w.start()

    def _on_simulation_done(self, results, summary, duration_hours, step_minutes):
        summary_text = print_simulation_summary(summary, console_output=False)
        for line in summary_text.split('\n'):
            self.consoleTextEdit.append(line)

        self.consoleTextEdit.append("")

        track = [(r['lat'], r['lon']) for r in results if 'lat' in r]
        if track:
            self.map_pic.set_trajectory(track)
            last = results[-1]
            if 'lat' in last:
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
                self.Answ.setPlainText("No satellite loaded.")
            return

        if self._vega_mode == 'delta':
            self.Answ.setPlainText("Computing orbital corrections…")

        self._delta_suggest_worker = DeltaSuggestWorker(line0, line1, line2)
        self._delta_suggest_worker.result_ready.connect(self._on_delta_suggestions_ready)
        self._delta_suggest_worker.start()

    def _on_delta_suggestions_ready(self, data):
        """Save recommendations silently — shown only when user switches to delta tab."""
        self._delta_suggestions_data = data

    def _show_delta_suggestions(self):
        """Render saved delta suggestions into Answ."""
        data = self._delta_suggestions_data
        if data is None:
            self.Answ.setPlainText("Could not compute corrections.")
            return

        orbit_type = data.get("orbit_type", "?")
        altitude   = data.get("altitude_km", 0)
        urgency    = data.get("urgency", "—")

        rec_incl = float(data.get("inclination", 0))
        rec_raan = float(data.get("raan", 0))
        rec_ecc  = float(data.get("eccentricity", 0))
        rec_arg  = float(data.get("argument_perigee", 0))
        rec_mm   = float(data.get("mean_motion", 0))

        # current values for diff
        cur_incl = cur_raan = cur_ecc = cur_arg = cur_mm = None
        if self.sat_data:
            try:
                f = parse_tle_fields(self.sat_data['line1'], self.sat_data['line2'])
                cur_incl = float(f['inclination'])
                cur_raan = float(f['raan'])
                ecc_str  = f['eccentricity']
                cur_ecc  = float(ecc_str) if '.' in ecc_str else float("0." + ecc_str)
                cur_arg  = float(f['argument_perigee'])
                cur_mm   = float(f['mean_motion'])
            except Exception:
                pass

        def _diff(rec, cur, decimals=4, unit="°"):
            if cur is None:
                return ""
            d = rec - cur
            if abs(d) < 10**(-decimals):
                return "  ✓"
            return f"  ({'+' if d>0 else ''}{d:.{decimals}f}{unit})"

        def _tag(key):
            s = data.get(key, "")
            return " ⚠" if s == "needs correction" else ""

        lines = [
            f"Orbital Corrections  —  {orbit_type}  |  alt: {altitude:.0f} km",
            f"     Urgency: {urgency}",
            "",
            f"  Inclination   {rec_incl:>12.4f}°{_diff(rec_incl, cur_incl)}{_tag('incl_status')}",
            f"  RAAN          {rec_raan:>12.4f}°{_diff(rec_raan, cur_raan)}",
            f"  Eccentricity  {rec_ecc:>12.6f}{_diff(rec_ecc, cur_ecc, 6, '')}",
            f"  Arg. Perigee  {rec_arg:>12.4f}°{_diff(rec_arg, cur_arg)}",
            f"  Mean Motion   {rec_mm:>14.8f}{_diff(rec_mm, cur_mm, 6, '')} rev/day{_tag('mm_status')}",
            "",
        ]

        # station-keeping recommendations (translated from Russian inline)
        recs = data.get("recommendations", [])
        _ru_to_en = {
            "Орбита стабильна": "Orbit stable — no correction required.",
            "North-South":      "North-South correction recommended.",
            "East-West":        "East-West correction recommended.",
            "циркуляризации":   "Circularisation manoeuvre recommended.",
            "Дрейф":            "Longitude drift detected.",
            "Наклонение":       "Inclination deviation detected.",
            "Эксцентриситет":   "Eccentricity out of tolerance.",
        }
        for rec in recs:
            translated = rec
            for ru, en in _ru_to_en.items():
                if ru in rec:
                    translated = en
                    break
            lines.append(f"  {translated}")

        if recs:
            lines.append("")

        # delta-v budget
        dv_total = data.get("total_delta_v", 0)
        dv_status = data.get("budget_status", "")
        lines += [
            f"  Delta-V budget: {dv_total:.2f} m/s  ({dv_status})",
            f"    Inclination:  {data.get('dv_inclination', 0):.2f} m/s",
            f"    Eccentricity: {data.get('dv_eccentricity', 0):.2f} m/s",
            f"    Drift:        {data.get('dv_drift', 0):.2f} m/s",
            "",
            "  Type 'delta' to apply these values.",
        ]

        self.Answ.setPlainText("\n".join(lines))

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
            elif self.sat_data:
                # AI module unavailable — show current TLE values as reference
                from tle_loader import parse_tle_fields
                try:
                    fields = parse_tle_fields(self.sat_data['line1'], self.sat_data['line2'])
                    self.Answ.setPlainText(
                        "[DELTA] Current TLE values (AI module unavailable):\n\n"
                        f"Inclination:      {fields['inclination']}°\n"
                        f"RAAN:             {fields['raan']}°\n"
                        f"Eccentricity:     {fields['eccentricity']}\n"
                        f"Arg. Perigee:     {fields['argument_perigee']}°\n"
                        f"Mean Motion:      {fields['mean_motion']} rev/day\n\n"
                        "Enter these or adjusted values when prompted."
                    )
                except Exception:
                    self.Answ.setPlainText("[DELTA] Enter orbit parameters when prompted.")
            else:
                self.Answ.setPlainText(
                    "[AI] No satellite loaded.\n"
                    "Use: sat <NORAD_ID>"
                )
        else:
            self._vega_mode = 'general'
            self.toggleModeButton.setText("⇄ DELTA VALUES")
            if self._ai_general_text:
                self.Answ.clear()
                self.Answ.append(self._ai_general_text)
            else:
                self.Answ.setPlainText("[AI] General analysis not available.")

    def update_satellite_marker(self, lat, lon):
        self.map_pic.set_satellite(lat, lon)


if __name__ == "__main__":
    app  = QtWidgets.QApplication(sys.argv)
    import platform
    if platform.system() == 'Windows':
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Sput.TLE.App")
    app.setWindowIcon(QIcon(os.path.join(os.path.dirname(os.path.abspath(__file__)), "App.ico")))

    # splash screen
    splash = SplashScreen()
    splash.show()
    splash.set_status("Loading TLE database")
    app.processEvents()

    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    window = MainWindow()
    window.hide()  # keep hidden until TLE loaded

    # finish splash when TLE loaded
    def _on_splash_done(success, count):
        splash.set_status(f"Ready — {count} satellites")
        app.processEvents()
        def _show_window():
            splash.stop()
            splash.close()
            window.consoleTextEdit.append("[SYSTEM] Console ready")
            window.consoleTextEdit.append("  Type 'help' for available commands\n")
            window.show()
            window._play_sound('start')
        QTimer.singleShot(800, _show_window)

    window._tle_loader.finished.connect(_on_splash_done)
    splash.raise_()

    with loop:
        loop.run_forever()
