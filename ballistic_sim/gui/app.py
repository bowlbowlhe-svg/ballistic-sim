"""基于 tkinter 的弹道模拟 GUI 主应用。"""

from __future__ import annotations

import queue
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, Optional

from ballistic_sim.cli import _compute_summary
from ballistic_sim.config import MonteCarloConfig, SimConfig
from ballistic_sim.gui.builder import (
    build_config_and_phases,
    default_preset,
    dump_config_to_yaml,
    list_presets,
    load_config_from_yaml,
    submodel_classes,
)
from ballistic_sim.gui.fields import build_form, read_model_variables, update_model_variables
from ballistic_sim.gui.runner import SimulationRunner
from ballistic_sim.phases.builder import build_phases
from ballistic_sim.simulator import SimResult
from ballistic_sim.viz import attach_launch_lla
from ballistic_sim.viz.profile import plot_altitude_range


class BallisticGuiApp(ttk.Frame):
    """弹道模拟 GUI 主窗口内容面板。"""

    def __init__(self, master: tk.Tk, *, mission: str = "projectile") -> None:
        super().__init__(master)
        self.master = master
        self.master.title("Ballistic Sim GUI")
        self.master.geometry("1200x800")

        self._runner = SimulationRunner()
        self._poll_id: Optional[str] = None

        self._build_toolbar()
        self._build_paned_body()

        self._current_mission = mission
        self._current_preset: Optional[str] = default_preset(mission)
        self._mission_var.set(mission)
        self._preset_var.set(self._current_preset or "")
        self._update_preset_list()
        self._load_config_to_forms()

    # -------------------------------------------------------------------------
    # 界面构造
    # -------------------------------------------------------------------------
    def _build_toolbar(self) -> None:
        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=4, pady=4)

        ttk.Label(toolbar, text="任务:").pack(side="left", padx=(0, 4))
        self._mission_var = tk.StringVar(value="projectile")
        mission_combo = ttk.Combobox(
            toolbar,
            textvariable=self._mission_var,
            values=["projectile", "missile", "rocket", "icbm", "suborbital"],
            state="readonly",
            width=12,
        )
        mission_combo.pack(side="left", padx=(0, 12))
        mission_combo.bind("<<ComboboxSelected>>", self._on_mission_changed)

        ttk.Label(toolbar, text="预设:").pack(side="left", padx=(0, 4))
        self._preset_var = tk.StringVar()
        self._preset_combo = ttk.Combobox(
            toolbar, textvariable=self._preset_var, state="readonly", width=14
        )
        self._preset_combo.pack(side="left", padx=(0, 12))
        self._preset_combo.bind("<<ComboboxSelected>>", self._on_preset_changed)

        ttk.Button(toolbar, text="Run", command=self._on_run).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Load YAML", command=self._on_load_yaml).pack(side="left", padx=4)
        ttk.Button(toolbar, text="Save YAML", command=self._on_save_yaml).pack(side="left", padx=4)

    def _build_paned_body(self) -> None:
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=4, pady=4)

        self._notebook = ttk.Notebook(paned)
        self._form_frames: Dict[str, ttk.Frame] = {}
        self._form_vars: Dict[str, Dict[str, tk.Variable]] = {}
        for name in ["vehicle", "launch", "environment", "guidance", "options"]:
            frame = ttk.Frame(self._notebook)
            inner = ttk.Frame(frame)
            inner.pack(fill="both", expand=True, padx=4, pady=4)
            self._notebook.add(frame, text=name.capitalize())
            self._form_frames[name] = inner

        # Monte Carlo 是 OptionsConfig 的子模型，单独一页
        mc_frame = ttk.Frame(self._notebook)
        mc_inner = ttk.Frame(mc_frame)
        mc_inner.pack(fill="both", expand=True, padx=4, pady=4)
        self._notebook.add(mc_frame, text="Monte Carlo")
        self._form_frames["monte_carlo"] = mc_inner

        paned.add(self._notebook, weight=1)

        right = ttk.Frame(paned)
        paned.add(right, weight=2)

        self._summary_text = tk.Text(right, wrap="word", height=10)
        self._summary_text.pack(fill="x", padx=4, pady=4)

        import matplotlib

        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        self._fig = Figure(figsize=(8, 6), dpi=100)
        self._ax = self._fig.add_subplot(111)
        self._canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

    # -------------------------------------------------------------------------
    # 事件处理
    # -------------------------------------------------------------------------
    def _on_mission_changed(self, _event: Any = None) -> None:
        mission = self._mission_var.get()
        self._current_mission = mission
        self._current_preset = default_preset(mission)
        self._update_preset_list()
        self._preset_var.set(self._current_preset or "")
        self._load_config_to_forms()

    def _on_preset_changed(self, _event: Any = None) -> None:
        self._current_preset = self._preset_var.get() or None
        self._load_config_to_forms()

    def _update_preset_list(self) -> None:
        presets = list_presets(self._current_mission)
        self._preset_combo["values"] = presets
        if presets and self._current_preset not in presets:
            self._current_preset = presets[0]
            self._preset_var.set(self._current_preset)

    # -------------------------------------------------------------------------
    # 配置读写
    # -------------------------------------------------------------------------
    def _load_config_to_forms(self) -> None:
        cfg, _ = build_config_and_phases(self._current_mission, self._current_preset)
        self._build_forms_if_needed(cfg)
        self._update_forms_from_config(cfg)

    def _build_forms_if_needed(self, cfg: SimConfig) -> None:
        classes = submodel_classes()
        for name, model_cls in classes.items():
            if name not in self._form_vars:
                self._form_vars[name] = build_form(self._form_frames[name], model_cls)

        mc_cfg = cfg.options.monte_carlo or MonteCarloConfig()
        if "monte_carlo" not in self._form_vars:
            self._form_vars["monte_carlo"] = build_form(self._form_frames["monte_carlo"], mc_cfg)

    def _update_forms_from_config(self, cfg: SimConfig) -> None:
        for name, model in [
            ("vehicle", cfg.vehicle),
            ("launch", cfg.launch),
            ("environment", cfg.environment),
            ("guidance", cfg.guidance),
            ("options", cfg.options),
        ]:
            update_model_variables(self._form_vars[name], model)

        mc_cfg = cfg.options.monte_carlo or MonteCarloConfig()
        update_model_variables(self._form_vars["monte_carlo"], mc_cfg)

    def _read_config_from_forms(self) -> SimConfig:
        classes = submodel_classes()
        kwargs: Dict[str, Any] = {"mission": self._current_mission}
        for name, model_cls in classes.items():
            vars_ = self._form_vars[name]
            data = read_model_variables(model_cls, vars_)
            kwargs[name] = model_cls(**data)

        mc_vars = self._form_vars["monte_carlo"]
        mc_data = read_model_variables(MonteCarloConfig, mc_vars)
        kwargs["options"] = kwargs["options"].model_copy(
            update={"monte_carlo": MonteCarloConfig(**mc_data)}
        )

        return SimConfig(**kwargs)

    # -------------------------------------------------------------------------
    # 运行仿真
    # -------------------------------------------------------------------------
    def _on_run(self) -> None:
        if self._runner.is_alive():
            messagebox.showwarning("运行中", "仿真已在后台运行，请等待完成。")
            return

        try:
            cfg = self._read_config_from_forms()
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("配置错误", f"读取参数失败: {exc}")
            return

        try:
            phases = build_phases(cfg)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("阶段构造错误", f"无法构造飞行阶段: {exc}")
            return

        self._summary_text.delete("1.0", "end")
        self._summary_text.insert("end", "正在运行仿真...\n")
        self._runner.run(cfg, phases)
        self._schedule_poll(cfg)

    def _schedule_poll(self, cfg: SimConfig) -> None:
        if self._poll_id is not None:
            try:
                self.after_cancel(self._poll_id)
            except Exception:  # noqa: BLE001
                pass

        def _poll() -> None:
            try:
                kind, payload = self._runner.result_queue.get_nowait()
            except queue.Empty:
                self._poll_id = self.after(100, _poll)
                return

            if kind == "ok":
                result = payload
                self._display_result(cfg, result)
            else:
                messagebox.showerror("仿真错误", f"运行失败: {payload}")
                self._summary_text.insert("end", f"错误: {payload}\n")
            self._poll_id = None

        self._poll_id = self.after(100, _poll)

    def _display_result(self, cfg: SimConfig, result: SimResult) -> None:
        attach_launch_lla(result, cfg.launch.lat_deg, cfg.launch.lon_deg, cfg.launch.alt_m)
        summary = _compute_summary(cfg, result)

        self._summary_text.delete("1.0", "end")
        self._summary_text.insert("end", f"Mission: {cfg.mission}\n")
        self._summary_text.insert("end", f"Stop   : {summary.get('stop_reason')}\n")
        self._summary_text.insert("end", f"TOF    : {summary.get('t_end_s')} s\n")
        if "range_m" in summary:
            self._summary_text.insert("end", f"Range  : {summary['range_m'] / 1e3:.2f} km\n")
        if "v_end_m_s" in summary:
            self._summary_text.insert("end", f"V_end  : {summary['v_end_m_s']:.1f} m/s\n")

        self._ax.clear()
        if result.y.size:
            try:
                fig = plot_altitude_range(result)
                self._ax.clear()
                for line in fig.axes[0].get_lines():
                    self._ax.plot(line.get_xdata(), line.get_ydata(), label=line.get_label())
                self._ax.set_xlabel("Downrange (km)")
                self._ax.set_ylabel("Altitude (km)")
                self._ax.set_title("Altitude vs Downrange")
                self._ax.set_ylim(bottom=0.0)
                self._ax.legend()
                self._ax.grid(True)
            except Exception as exc:  # noqa: BLE001
                self._ax.text(0.5, 0.5, f"绘图失败: {exc}", ha="center", va="center")
        else:
            self._ax.text(0.5, 0.5, "无轨迹数据", ha="center", va="center")
        self._fig.tight_layout()
        self._canvas.draw()

    # -------------------------------------------------------------------------
    # YAML 加载/保存
    # -------------------------------------------------------------------------
    def _on_load_yaml(self) -> None:
        path_str = filedialog.askopenfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not path_str:
            return
        try:
            cfg = load_config_from_yaml(Path(path_str))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("加载失败", f"无法加载 YAML: {exc}")
            return

        self._current_mission = cfg.mission
        self._mission_var.set(cfg.mission)
        self._current_preset = None
        self._update_preset_list()
        self._preset_var.set("")
        self._build_forms_if_needed(cfg)
        self._update_forms_from_config(cfg)

    def _on_save_yaml(self) -> None:
        path_str = filedialog.asksaveasfilename(
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        if not path_str:
            return
        try:
            cfg = self._read_config_from_forms()
            dump_config_to_yaml(cfg, Path(path_str))
            messagebox.showinfo("保存成功", f"配置已保存到 {path_str}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("保存失败", f"无法保存 YAML: {exc}")

    # -------------------------------------------------------------------------
    # 公共属性
    # -------------------------------------------------------------------------
    @property
    def current_mission(self) -> str:
        return self._current_mission

    @property
    def current_preset(self) -> Optional[str]:
        return self._current_preset
