"""Radar configuration: parameters, derived quantities, validation and `mmwc` command generation.

Every rule here comes from the vendor docs or legacy experience. Each rule is labelled in
its ``Issue`` so that a failing configuration explains itself. Rules marked ``assumption``
are educated guesses that still need bench confirmation. When the chip disagrees with a rule,
fix the rule and record the evidence in ``docs/decisions/DECISIONS.md``.

Key vendor formulas (RS6x_7x mmWave Radar Reference Manual §3.2):

* ADC samples ``N = floor(range_mm / res_mm)``, allowed 32..1024; ADC is 10 MS/s complex,
  so the chirp sampling time is ``Ts = N * 0.1 us``.
* Range FFT length = next power of two >= N.
* Bandwidth ``B = c / (2 res)``, so the slope is ``S = B / Ts = c / (2 R_max 0.1 us)``. It
  depends only on ``range_mm`` and must stay <= 400 MHz/us (datasheet).
* 2D mode: intervals ``K = floor(2 v_max / v_res)`` (2..512), Doppler FFT length = next pow2,
  interval spacing ``Tc = lambda / (4 v_max)``.
* DataCube bytes = RangeFFT x DopplerFFT x Tx x Rx x 4; RS6130 stores 128 KB (256 KB with lossy 2x compression).
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

C = 299_792_458.0
ADC_FS_HZ = 10e6                  # complex sampling rate, fixed by the SDK
F_MIN_MHZ, F_MAX_MHZ = 58_000, 64_000
MAX_SLOPE_MHZ_PER_US = 400.0
DATACUBE_MAX_BYTES = 128 * 1024
DATACUBE_MAX_COMPRESSED = 256 * 1024
HIF_MAX_PAYLOAD = 4095
HIF_PAYLOAD_HDR = 12
HIF_PACKET_OVERHEAD = 1 + 1 + 4 + 4 + HIF_PAYLOAD_HDR   # magic, check8, header, check32, payload hdr
UART_BITS_PER_BYTE = 10
# SPI link (EXP-002). Vendor HIF Ref V1.2 §8.3.1 example: 256 kB in 85 ms (~3 MB/s effective, device
# side). Through the dev board's CH347T USB bridge each transfer is a USB round trip, so we assume
# 40 % of the raw clock rate, capped at the vendor figure. ASSUMPTION until linkbench measures it.
SPI_EFFICIENCY_ASSUMED = 0.4
SPI_VENDOR_EFF_BYTES_S = 3.0e6
RX_GAIN_RF_ALLOWED = (27, 33, 39, 45, 51)
RX_GAIN_DIG_ALLOWED = (-18, -12, -6, 0, 6, 12, 18)
TX_ATT_ALLOWED = (0, 1, 2, 3, 4, 6, 8)
DEFAULT_HPF_HZ = 0.5e6            # vendor default; numeric code -> Hz mapping of `mmwc hpf` is UNKNOWN
RX_SPACING_M = 2.45e-3
# Chirp timing model from the chip's own `chirp_info` (first bench run 2026-09-25, fw V4.3.1.0,
# range 6400/200 -> slope 234 MHz/us): the ramp starts 3.15 us (738 MHz) before the ADC window
# and continues ~0.5 us after it; the chirp repeats every ~14.25 us (Ts 3.2 us + ~11 us).
# ONE data point: treat as a model to be refined, not a spec.
CHIRP_PRE_US = 3.15
CHIRP_POST_US = 0.5
CHIRP_IDLE_US = 11.0


def next_pow2(n: int) -> int:
    return 1 << max(0, math.ceil(math.log2(max(1, n))))


@dataclass
class RadarConfig:
    """User-facing radar settings. ``None`` means "do not send; keep chip default"."""

    name: str = "custom"
    notes: str = ""
    # waveform
    fft_mode: int = 1                  # 1 = 1D (range only), 2 = 2D (range-Doppler / multi-interval)
    txrx_mode: int = 1                 # 1 = 1T3R
    freq_start_mhz: int = 58_000
    freq_max_mhz: int | None = None    # second arg of `mmwc freq`; semantics unverified
    range_max_mm: int = 6400
    range_res_mm: int = 200
    chirp_num: int = 4                 # on-chip complex chirp averaging (power of 2, <= 128)
    intv_period_us: int | None = None  # `mmwc intv <period_us> <num>` (legacy V3 used 200 1)
    intv_num: int = 1
    veloc_max_mm_s: int | None = None  # 2D only
    veloc_res_mm_s: int | None = None  # 2D only
    dop_fft: bool = False              # 2D only: run the Doppler FFT on chip
    frame_period_ms: int = 25
    # baseband
    rx_gain_rf: int | None = None
    rx_gain_dig: int | None = None
    hpf_code: int | None = None        # raw `mmwc hpf 255 <code>`; unit unknown (legacy used 16)
    hpf_hz_assumed: float | None = None  # what you *believe* hpf_code means (for derived numbers only)
    tx_rf_att_db: int | None = None
    clutter_rm: bool | None = False    # static clutter removal (vendor: disable on moving platform)
    dc_suppression: bool | None = None
    window_range: str | None = None    # "hann" | "rect"
    window_doppler: str | None = None
    power_save: bool | None = None     # `ps` (power save); None = do not send
    cmd_order: str = "p2"              # see RadarConfig.CMD_ORDERS
    # link / session
    link: str = "uart"                 # "uart" (dual-UART, report on UART1) | "spi" (report on SPI, P1812 only)
    spi_clock_hz: int = 30_000_000     # host SPI clock (CH347: 60/30/15/7.5/3.75 MHz ...); device slave max 56 MHz
    baudrate: int = 1_000_000          # vendor tools recommend 1 Mbaud on the data UART; 2 Mbaud is the max
    ctrl_baudrate: int = 115_200
    send_open: bool = True
    extra_commands: list[str] = field(default_factory=list)   # raw, sent after baseband config
    # decoding conventions (to test README D1/D3 without code changes)
    iq_order: str = "imag_low"         # HIF manual: imag = low 16 bits. Alternative: "real_low"
    cube_order: str = "tx_rx_dop_range"  # alternative: "dop_tx_rx_range"
    n_rx: int = 3
    n_tx: int = 1

    # ------------------------------------------------------------------ io
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "RadarConfig":
        names = {f.name for f in fields(cls)}
        unknown = set(d) - names
        cfg = cls(**{k: v for k, v in d.items() if k in names})
        if unknown:
            cfg.notes = (cfg.notes + f" [ignored unknown keys: {sorted(unknown)}]").strip()
        return cfg

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "RadarConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def copy(self, **changes) -> "RadarConfig":
        c = copy.deepcopy(self)
        for k, v in changes.items():
            setattr(c, k, v)
        return c

    # ------------------------------------------------------------ commands
    def _blocks(self) -> dict[str, list[str]]:
        B: dict[str, list[str]] = {}
        B["open"] = ["open"] if self.send_open else []
        B["ps"] = [f"ps {1 if self.power_save else 0}"] if self.power_save is not None else []
        B["mode"] = [f"mode {self.txrx_mode} {self.fft_mode}"]
        B["range"] = [f"range {self.range_max_mm} {self.range_res_mm}"]
        B["veloc"] = ([f"veloc {self.veloc_max_mm_s} {self.veloc_res_mm_s}"]
                      if self.fft_mode == 2 and self.veloc_max_mm_s and self.veloc_res_mm_s else [])
        B["freq"] = [f"freq {self.freq_start_mhz} {self.freq_max_mhz}" if self.freq_max_mhz is not None
                     else f"freq {self.freq_start_mhz}"]
        B["chirp_num"] = [f"chirp_num {self.chirp_num}"]
        B["intv"] = [f"intv {self.intv_period_us} {self.intv_num}"] if self.intv_period_us is not None else []
        B["frame"] = [f"frame {self.frame_period_ms} 0"]
        bb = []
        if self.rx_gain_rf is not None:
            bb.append(f"rx_gain rf 255 {self.rx_gain_rf}")
        if self.rx_gain_dig is not None:
            bb.append(f"rx_gain dig 255 {self.rx_gain_dig}")
        if self.hpf_code is not None:
            bb.append(f"hpf 255 {self.hpf_code}")
        if self.tx_rf_att_db is not None:
            bb.append(f"tx_rf_att {self.tx_rf_att_db}")
        if self.dc_suppression is not None:
            bb.append(f"dc_suppression {1 if self.dc_suppression else 0}")
        if self.window_range is not None:
            bb.append(f"win 0 {1 if self.window_range == 'hann' else 0}")
        if self.window_doppler is not None:
            bb.append(f"win 1 {1 if self.window_doppler == 'hann' else 0}")
        B["baseband"] = bb
        B["clutter"] = [f"clutter_rm {1 if self.clutter_rm else 0} 0"] if self.clutter_rm is not None else []
        B["extra"] = [c.strip() for c in self.extra_commands if c.strip()]
        # The shell parses atoi(argv[1]): "on"/"off" both mean 0 = OFF (EXP-003, mmw_shell.c cmd_mmw_dop_fft).
        B["dop_fft"] = [f"dop_fft {1 if (self.fft_mode == 2 and self.dop_fft) else 0}"]
        B["report"] = (["comtype 0", f"baudrate {self.baudrate}", "report cube 4294967295", "comtype 1",
                        "report cube 4294967295"] if self.link == "spi" else ["report cube 4294967295"])
        # `report` re-initialises the HIF with the com type / rate set BEFORE it (mmw_shell.c, D-008).
        # comtype is always sent explicitly so UART<->SPI switches work within one power cycle.
        # For SPI the HIF is forced through a UART->SPI re-init (comtype 0 + report, then comtype 1): if the
        # chip is already in SPI mode, `report` would NOT re-init it, and a wedged SPI HIF (bench
        # 2026-09-25, run 205412) then survives into the next session.
        # The UART step needs an explicit baudrate: with com_rate = 0 the chip answers `report` with
        # "host interface init fail" (bench 2026-09-25 spi-probe; same message as EXP-001 attempts 1-2,
        # where `report` also came before `baudrate`).
        # The UART->SPI re-init happens in the report block, i.e. AFTER mode/range/frame are set:
        # a `report` issued while the chip still holds the previous run's frame period (5 ms) together
        # with the new 2D mode hung the chip and the USB bridge re-enumerated (linkbench 2026-09-25).
        # UART: exactly the order that streamed on 2026-09-25 (`baudrate` only). Adding `comtype 0` here made
        # the chip stop answering at `report` and the USB bridge re-enumerate (linkbench 21:42) - the module
        # firmware is not the SDK build, and its `comtype` may do more than set com_mode.
        B["baud"] = [] if self.link == "spi" else [f"baudrate {self.baudrate}"]
        B["uart"] = ["uart on"]
        return B

    #: Command orders. "p2" = the order used by legacy perception2, which is the only host-side
    #: tool known to have received cube data from a dev board over two UARTs. "v3" = legacy V3 firmware
    #: order (MCU host). "baseband_last" = the first ravidata order (baseband after frame,
    #: report/baud/uart last); on 2026-09-25 it gave "host interface init fail" on `report cube`
    #: and no data.
    CMD_ORDERS = {
        "p2": ["open", "mode", "baud", "chirp_num", "freq", "range", "veloc", "clutter", "dop_fft", "frame",
               "baseband", "extra", "report", "uart"],
        "v3": ["mode", "ps", "range", "freq", "frame", "intv", "report", "chirp_num", "baud", "uart", "dop_fft",
               "baseband", "clutter", "extra"],
        "baseband_last": ["open", "ps", "mode", "range", "veloc", "freq", "chirp_num", "intv", "frame", "baseband",
                          "clutter", "extra", "dop_fft", "report", "baud", "uart"],
    }

    def to_commands(self) -> list[str]:
        """Ordered `mmwc` command list (without the ``mmwc`` prefix), per ``cmd_order``.

        The order matters on this chip (see CMD_ORDERS). ``stop`` always comes first.
        """
        B = self._blocks()
        order = self.CMD_ORDERS.get(self.cmd_order, self.CMD_ORDERS["p2"])
        cmds = ["stop"]
        for name in order:
            cmds.extend(B[name])
        return cmds

    # Read-back queries sent after configuring (their answers go into the run log).
    # NOT here on purpose: `comtype` and `baudrate`. In the SDK V2.1.2 shell source (mmw_shell.c) both are
    # SETTERS that do `atoi(argv[1])` without checking argc, so a bare "query" writes com_mode/com_rate = 0
    # (com_rate 0 = firmware default rate) and the next `report` re-initialises the HIF with it (D-008).
    # `report get` is the read-only form of `report` (argc == 1 / "get" branch).
    QUERY_COMMANDS = ("version", "state", "report get", "chirp_info", "fft_len", "range", "freq", "frame",
                      "intv", "getreportcfg", "dop_fft")
    # bare `dop_fft` is a pure getter in mmw_shell.c (argc == 1 prints "dop fft=%d"); do NOT send "dop_fft get"
    # to an unknown firmware, an older parser could atoi("get") = 0 and switch the FFT off.

    # ------------------------------------------------------------- derive
    def derive(self) -> "Derived":
        return derive(self)

    def validate(self) -> list["Issue"]:
        return validate(self)


@dataclass
class Derived:
    n_samples: int
    range_fft_len: int
    bin_mm: float                   # spacing of range FFT bins (zero padding accounted for)
    bandwidth_hz: float
    ts_us: float
    slope_mhz_per_us: float
    freq_end_mhz: float            # end of the *sampled* (valid) part of the sweep
    valid_start_mhz: float          # model: start + slope * CHIRP_PRE_US
    sweep_end_mhz: float            # model: whole ramp incl. settling (what must stay <= 64 GHz)
    chirp_period_us: float          # model: Ts + CHIRP_IDLE_US
    wavelength_mm: float
    d_over_lambda: float
    beat_hz_per_m: float
    n_intervals: int                # K (1 in 1D mode)
    doppler_len: int                # length of the Doppler/interval axis in the cube
    interval_spacing_us: float | None
    v_res_mm_s: float | None
    active_time_ms: float           # rough time the chirps occupy per frame
    cube_bytes: int
    cube_dwords: int
    packets_per_frame: int
    wire_bytes_per_frame: int
    tx_time_ms: float               # UART time to ship one frame (our wire model: 10 bit/byte + HIF overhead)
    tx_time_vendor_ms: float        # vendor formula (HIF Ref V1.2 §8.3.1): 48000*N_ant*bins*doppler/baud ms
    max_fps_link: float
    link_utilisation: float         # tx_time / frame_period
    expected_fps: float
    hpf_corner_range_m: float       # range where the IF HPF corner sits (with assumed HPF)
    usable_range_m: float           # 0.8 * R_max (vendor: last 20 % unusable)
    avg_loss_db_1ms: float          # chirp-averaging loss at 1.0 m/s (optimistic chirp spacing)
    avg_loss_db_1ms_pess: float | None  # same with pessimistic spacing (= intv period), if known

    def to_dict(self) -> dict:
        return asdict(self)

    def summary_lines(self) -> list[str]:
        L = [
            f"ADC samples N={self.n_samples}, range FFT {self.range_fft_len}, bin {self.bin_mm:.1f} mm, "
            f"usable to {self.usable_range_m:.2f} m",
            f"valid sweep B={self.bandwidth_hz/1e9:.3f} GHz, Ts={self.ts_us:.1f} us, slope {self.slope_mhz_per_us:.0f} MHz/us; "
            f"model: valid {self.valid_start_mhz:.0f}-{self.freq_end_mhz:.0f} MHz, ramp ends ~{self.sweep_end_mhz:.0f} MHz, "
            f"chirp period ~{self.chirp_period_us:.1f} us; d/lambda={self.d_over_lambda:.3f}",
            f"cube: Doppler/interval axis {self.doppler_len} x 3 RX x {self.range_fft_len} = {self.cube_bytes} B "
            f"({self.packets_per_frame} HIF packets, {self.wire_bytes_per_frame} B on wire)",
            f"link: {self.tx_time_ms:.1f} ms/frame on the link (UART vendor formula {self.tx_time_vendor_ms:.0f} ms) -> max {self.max_fps_link:.1f} fps; "
            f"utilisation {100*self.link_utilisation:.0f} % at {self.expected_fps:.1f} fps requested",
            f"HPF corner (assumed) at ~{self.hpf_corner_range_m:.2f} m range; beat {self.beat_hz_per_m/1e6:.2f} MHz per m",
        ]
        if self.n_intervals > 1:
            L.append(f"2D: K={self.n_intervals} intervals, spacing {self.interval_spacing_us:.0f} us, "
                     f"v_res {self.v_res_mm_s:.0f} mm/s, active {self.active_time_ms:.1f} ms/frame")
        pess = f", pessimistic {self.avg_loss_db_1ms_pess:.1f} dB" if self.avg_loss_db_1ms_pess is not None else ""
        L.append(f"chirp-averaging loss at 1 m/s: optimistic {self.avg_loss_db_1ms:.1f} dB{pess} (assumption)")
        return L


@dataclass
class Issue:
    level: str        # "error" | "warning" | "info"
    code: str
    message: str
    source: str = ""  # where the rule comes from (doc section, legacy, assumption)

    def __str__(self) -> str:
        src = f" [{self.source}]" if self.source else ""
        return f"{self.level.upper():7s} {self.code}: {self.message}{src}"


def _avg_loss_db(n: int, spacing_s: float, v: float, lam_m: float) -> float:
    if n <= 1:
        return 0.0
    dphi = 4 * math.pi * v * spacing_s / lam_m
    s = math.sin(dphi / 2)
    if abs(s) < 1e-12:
        return 0.0
    g = abs(math.sin(n * dphi / 2) / (n * s))
    return -20 * math.log10(max(g, 1e-6))


def derive(cfg: RadarConfig) -> Derived:
    n = max(1, int(cfg.range_max_mm // max(1, cfg.range_res_mm)))
    nfft = next_pow2(n)
    res_m = cfg.range_res_mm / 1000.0
    B = C / (2 * res_m)
    ts_us = n * 0.1
    slope = (B / 1e6) / ts_us
    valid_start = cfg.freq_start_mhz + slope * CHIRP_PRE_US
    f_end = valid_start + B / 1e6
    sweep_end = f_end + slope * CHIRP_POST_US
    chirp_period = ts_us + CHIRP_IDLE_US
    fc = (valid_start * 1e6 + B / 2)
    lam = C / fc
    bin_mm = cfg.range_res_mm * n / nfft
    rmax_m = cfg.range_max_mm / 1000.0
    beat_per_m = ADC_FS_HZ / rmax_m

    if cfg.fft_mode == 2 and cfg.veloc_max_mm_s and cfg.veloc_res_mm_s:
        K = max(1, int(2 * cfg.veloc_max_mm_s // cfg.veloc_res_mm_s))
        tc_us = (lam / (4 * cfg.veloc_max_mm_s / 1000.0)) * 1e6
        v_res = cfg.veloc_res_mm_s
        dlen = next_pow2(K) if cfg.dop_fft else K
        active_ms = tc_us * K / 1000.0
    else:
        K, tc_us, v_res, dlen = 1, None, None, 1
        active_ms = cfg.chirp_num * chirp_period / 1000.0

    dwords = nfft * dlen * cfg.n_tx * cfg.n_rx
    cube_bytes = dwords * 4
    per_pkt = HIF_MAX_PAYLOAD - HIF_PAYLOAD_HDR
    per_pkt -= per_pkt % 4
    pkts = max(1, math.ceil(cube_bytes / per_pkt))
    wire = cube_bytes + pkts * HIF_PACKET_OVERHEAD
    if cfg.link == "spi":
        spi_bps = min(SPI_EFFICIENCY_ASSUMED * cfg.spi_clock_hz / 8.0, SPI_VENDOR_EFF_BYTES_S)
        tx_ms = wire / spi_bps * 1000.0
    else:
        tx_ms = wire * UART_BITS_PER_BYTE / cfg.baudrate * 1000.0
    # Vendor rule of thumb (HIF Reference Manual V1.2 §8.3.1, UART): t_report = ceil(48000*N*Rbins*Vbins/baud) ms,
    # i.e. ~48 bit-times per complex bin (ours: ~40 + headers). Kept side by side, not merged.
    tx_vendor_ms = float(math.ceil(48000 * cfg.n_tx * cfg.n_rx * nfft * dlen / cfg.baudrate))
    fps_req = 1000.0 / max(1, cfg.frame_period_ms)
    hpf_hz = cfg.hpf_hz_assumed if cfg.hpf_hz_assumed else DEFAULT_HPF_HZ

    opt_spacing = chirp_period * 1e-6
    loss_opt = _avg_loss_db(cfg.chirp_num, opt_spacing, 1.0, lam)
    loss_pess = None
    if cfg.fft_mode == 1 and cfg.intv_period_us:
        loss_pess = _avg_loss_db(cfg.chirp_num, cfg.intv_period_us * 1e-6, 1.0, lam)

    return Derived(
        n_samples=n, range_fft_len=nfft, bin_mm=bin_mm, bandwidth_hz=B, ts_us=ts_us,
        slope_mhz_per_us=slope, freq_end_mhz=f_end, valid_start_mhz=valid_start, sweep_end_mhz=sweep_end,
        chirp_period_us=chirp_period, wavelength_mm=lam * 1000,
        d_over_lambda=RX_SPACING_M / lam, beat_hz_per_m=beat_per_m, n_intervals=K,
        doppler_len=dlen, interval_spacing_us=tc_us, v_res_mm_s=v_res, active_time_ms=active_ms,
        cube_bytes=cube_bytes, cube_dwords=dwords, packets_per_frame=pkts, wire_bytes_per_frame=wire,
        tx_time_ms=tx_ms, tx_time_vendor_ms=tx_vendor_ms, max_fps_link=1000.0 / tx_ms, link_utilisation=tx_ms * fps_req / 1000.0,
        expected_fps=fps_req, hpf_corner_range_m=hpf_hz / beat_per_m, usable_range_m=0.8 * rmax_m,
        avg_loss_db_1ms=loss_opt, avg_loss_db_1ms_pess=loss_pess,
    )


def validate(cfg: RadarConfig) -> list[Issue]:
    I: list[Issue] = []
    d = derive(cfg)
    E = lambda c, m, s="": I.append(Issue("error", c, m, s))      # noqa: E731
    W = lambda c, m, s="": I.append(Issue("warning", c, m, s))    # noqa: E731
    N = lambda c, m, s="": I.append(Issue("info", c, m, s))       # noqa: E731

    if cfg.cmd_order not in RadarConfig.CMD_ORDERS:
        E("cmd_order", f"unknown cmd_order {cfg.cmd_order!r}; use one of {list(RadarConfig.CMD_ORDERS)}")
    if cfg.fft_mode not in (1, 2):
        E("mode", f"fft_mode must be 1 or 2, got {cfg.fft_mode}")
    if not 32 <= d.n_samples <= 1024:
        E("adc_samples", f"range/res = {d.n_samples} ADC samples; allowed 32..1024 "
          f"(e.g. for {cfg.range_res_mm} mm bins use range >= {32*cfg.range_res_mm} mm)", "Radar Ref §3.2.2")
    if d.n_samples != d.range_fft_len:
        W("adc_not_pow2", f"{d.n_samples} samples are zero-padded to {d.range_fft_len}; vendor recommends "
          "range/res to be a power of 2", "Radar Ref §3.2.8")
    if d.slope_mhz_per_us > MAX_SLOPE_MHZ_PER_US:
        E("slope", f"chirp slope {d.slope_mhz_per_us:.0f} MHz/us > {MAX_SLOPE_MHZ_PER_US:.0f} max. Slope depends only on "
          f"range_mm: need range_mm >= {C/(2*MAX_SLOPE_MHZ_PER_US*1e12*1e-7)*1000:.0f}", "datasheet §6.4 + Radar Ref §3.2.2")
    if cfg.freq_start_mhz < F_MIN_MHZ:
        E("freq_low", f"start {cfg.freq_start_mhz} MHz < {F_MIN_MHZ}", "datasheet")
    if d.sweep_end_mhz > F_MAX_MHZ:
        max_b = F_MAX_MHZ - 58_000 - d.slope_mhz_per_us * (CHIRP_PRE_US + CHIRP_POST_US)
        E("freq_high", f"ramp ends at ~{d.sweep_end_mhz:.0f} MHz > {F_MAX_MHZ} (valid B={d.bandwidth_hz/1e9:.2f} GHz plus "
          f"~{d.slope_mhz_per_us*(CHIRP_PRE_US+CHIRP_POST_US):.0f} MHz settling at this slope). Max valid B from 58 GHz "
          f"is ~{max_b/1000:.2f} GHz -> range_res >= {C/(2*max_b*1e6)*1000:.0f} mm, or lower the start",
          "datasheet + chirp_info model (1 measurement)")
    if cfg.freq_max_mhz not in (None, 0) and cfg.freq_max_mhz > F_MAX_MHZ:
        W("freq_max_arg", f"freq max argument {cfg.freq_max_mhz} > {F_MAX_MHZ}; meaning of this argument is unverified",
          "legacy V3 used 69000")
    if cfg.range_max_mm < 5000:
        W("rmax_lt_5m", f"R_max {cfg.range_max_mm/1000:.2f} m < 5 m: vendor advises > 5 m; objects beyond R_max may "
          "alias into the nearest bins", "Radar Ref §3.2.8, §3.3.4")
    if cfg.chirp_num < 1 or cfg.chirp_num > 128 or (cfg.chirp_num & (cfg.chirp_num - 1)):
        E("chirp_num", f"chirp_num {cfg.chirp_num} must be a power of 2 in 1..128", "Radar Ref §3.2.4")
    if d.avg_loss_db_1ms > 3.0:
        W("avg_loss", f"estimated chirp-averaging loss {d.avg_loss_db_1ms:.1f} dB at 1 m/s", "assumption")
    if d.avg_loss_db_1ms_pess is not None and d.avg_loss_db_1ms_pess > 3.0:
        W("avg_loss_pess", f"if chirps are spread over intv_period_us={cfg.intv_period_us}, averaging "
          f"{cfg.chirp_num} chirps loses ~{d.avg_loss_db_1ms_pess:.1f} dB at 1 m/s", "assumption (research D8)")
    if cfg.fft_mode == 2:
        if not (cfg.veloc_max_mm_s and cfg.veloc_res_mm_s):
            E("veloc_missing", "2D mode needs veloc_max_mm_s and veloc_res_mm_s")
        else:
            if not 2 <= d.n_intervals <= 512:
                E("intervals", f"K={d.n_intervals} intervals, allowed 2..512", "Radar Ref §3.2.3")
            if d.doppler_len != d.n_intervals and cfg.dop_fft:
                W("k_not_pow2", f"K={d.n_intervals} zero-padded to {d.doppler_len}", "Radar Ref §3.2.8")
            if cfg.dop_fft and d.n_intervals == 2:
                E("dop2", "Doppler FFT with 2 points is not supported; set dop_fft off", "Radar Ref §3.2.7.2")
            chirp_time_us = cfg.chirp_num * d.chirp_period_us
            if d.interval_spacing_us is not None and chirp_time_us > d.interval_spacing_us:
                E("interval_too_short", f"{cfg.chirp_num} chirps x ~{d.chirp_period_us:.1f} us = {chirp_time_us:.0f} us do not fit in "
                  f"the interval spacing {d.interval_spacing_us:.0f} us (set by v_max). Lower chirp_num, v_max or range",
                  "Radar Ref §3.2.3 + assumption on chirp idle")
            if cfg.veloc_max_mm_s < 1800:
                W("vmax_low", f"v_max {cfg.veloc_max_mm_s} mm/s < 1800: walking + sweep will alias", "research D8")
            if d.active_time_ms > cfg.frame_period_ms:
                E("frame_too_short", f"chirps need ~{d.active_time_ms:.1f} ms > frame period {cfg.frame_period_ms} ms")
    else:
        if cfg.dop_fft:
            W("dop_fft_1d", "dop_fft ignored in 1D mode")
    if d.cube_bytes > DATACUBE_MAX_BYTES:
        if d.cube_bytes <= DATACUBE_MAX_COMPRESSED:
            W("cube_compressed", f"cube {d.cube_bytes} B > 128 KB: chip will use lossy 2x compression", "Radar Ref §3.2.8")
        else:
            E("cube_too_big", f"cube {d.cube_bytes} B > 256 KB limit", "Radar Ref §3.2.8")
    if cfg.link not in ("uart", "spi"):
        E("link_kind", f"link must be 'uart' or 'spi', got {cfg.link!r}")
    if cfg.link == "spi":
        N("spi_assumed", f"SPI time per frame assumes {SPI_EFFICIENCY_ASSUMED:.0%} of {cfg.spi_clock_hz/1e6:.1f} MHz "
          f"(cap {SPI_VENDOR_EFF_BYTES_S/1e6:.1f} MB/s); P1812 only; the chip aborts a frame the host does not drain "
          "within ~1 frame period", "assumption + mmw_report.c")
        if cfg.spi_clock_hz > 56_000_000:
            W("spi_clock", f"SPI clock {cfg.spi_clock_hz/1e6:.0f} MHz > device slave default 56 MHz", "hif_config.h")
    if cfg.link == "uart" and cfg.baudrate > 2_000_000:
        E("baud", f"baudrate {cfg.baudrate} > 2 Mbaud max", "RadarAnalysisTool A.24")
    if d.link_utilisation > 1.0:
        E("link", f"frame needs {d.tx_time_ms:.1f} ms on the {cfg.link.upper()} but the period is {cfg.frame_period_ms} ms "
          f"(max {d.max_fps_link:.1f} fps). Raise frame_period_ms to >= {math.ceil(d.tx_time_ms*1.25)} "
          "or shrink the cube", "derived")
    elif cfg.link == "uart" and d.tx_time_vendor_ms > cfg.frame_period_ms:
        W("link_vendor", f"vendor UART formula gives {d.tx_time_vendor_ms:.0f} ms/frame > period "
          f"{cfg.frame_period_ms} ms (our model: {d.tx_time_ms:.1f} ms)", "HIF Ref V1.2 §8.3.1")
    elif d.link_utilisation > 0.75:
        W("link_tight", f"{cfg.link.upper()} {100*d.link_utilisation:.0f} % busy; frames may be dropped", "derived")
    if d.hpf_corner_range_m > 0.3:
        W("hpf_near", f"IF high-pass corner falls at ~{d.hpf_corner_range_m:.2f} m: closer targets are attenuated "
          f"(beat {d.beat_hz_per_m/1e6:.2f} MHz/m). Larger range_mm (lower slope) pushes it FARTHER, smaller range_mm closer", "Radar Ref §3.3.4 (+ HPF assumption)")
    if cfg.rx_gain_rf is not None and cfg.rx_gain_rf not in RX_GAIN_RF_ALLOWED:
        E("rx_gain_rf", f"rx_gain_rf {cfg.rx_gain_rf} not in {RX_GAIN_RF_ALLOWED}", "Radar Ref §3.3.1")
    if cfg.rx_gain_dig is not None and cfg.rx_gain_dig not in RX_GAIN_DIG_ALLOWED:
        E("rx_gain_dig", f"rx_gain_dig {cfg.rx_gain_dig} not in {RX_GAIN_DIG_ALLOWED}", "RadarAnalysisTool A.27")
    if cfg.tx_rf_att_db is not None and cfg.tx_rf_att_db not in TX_ATT_ALLOWED:
        E("tx_att", f"tx_rf_att {cfg.tx_rf_att_db} not in {TX_ATT_ALLOWED}")
    if cfg.hpf_code is not None and cfg.hpf_hz_assumed is None:
        N("hpf_unknown", f"hpf code {cfg.hpf_code} sent; its unit is unknown so derived HPF numbers assume 0.5 MHz")
    if cfg.clutter_rm:
        W("clutter_rm", "static clutter removal ON: static obstacles of a stopped user can vanish", "Radar Ref §3.3.6")
    if cfg.iq_order != "imag_low":
        N("iq_order", "decoding with real in low 16 bits (legacy V3 convention, contradicts HIF manual)")
    if cfg.frame_period_ms < 10:
        W("frame_fast", f"frame period {cfg.frame_period_ms} ms is very short")
    return I


def has_errors(issues: list[Issue]) -> bool:
    return any(i.level == "error" for i in issues)


# --------------------------------------------------------------------------- presets
def presets() -> dict[str, RadarConfig]:
    """Starting points. None of them is validated on hardware yet."""
    P = {}
    P["coarse-1d-20cm"] = RadarConfig(
        name="coarse-1d-20cm", notes="Coarse 1D: 20 cm bins, 32 samples, tiny cube, fast frames.",
        fft_mode=1, freq_start_mhz=58_000, range_max_mm=6400, range_res_mm=200, chirp_num=4,
        frame_period_ms=25)
    P["coarse-1d-30cm"] = RadarConfig(
        name="coarse-1d-30cm", notes="Owner's 'rough 30 cm is enough' case: N=32, B=0.5 GHz.",
        fft_mode=1, freq_start_mhz=58_000, range_max_mm=9600, range_res_mm=300, chirp_num=4,
        frame_period_ms=25)
    P["legal-1d-40mm"] = RadarConfig(
        name="legal-1d-40mm", notes="Fine 1D within the slope limit: N=128, B=3.75 GHz.",
        fft_mode=1, freq_start_mhz=58_000, range_max_mm=5120, range_res_mm=40, chirp_num=4,
        frame_period_ms=25)
    P["legal-1d-30mm"] = RadarConfig(
        name="legal-1d-30mm", notes="Near the widest sweep that fits 58-64 GHz incl. settling (N=256, B=5.0 GHz).",
        fft_mode=1, freq_start_mhz=58_000, range_max_mm=7680, range_res_mm=30, chirp_num=4,
        frame_period_ms=40)
    # 2D presets (2026-09-25 evening): the first runs with the fixed `dop_fft 1` AND `baudrate 2000000` sent
    # 0 bytes. The only 2D path proven on hardware is 1 Mbaud + chip Doppler off (linkbench). So: 1 Mbaud and a
    # link-feasible period by default; the chip-Doppler and 2 Mbaud variants are separate presets to A/B.
    P["slowtime-2d-coarse"] = RadarConfig(
        name="slowtime-2d-coarse", notes="2D, chip Doppler OFF (slow time, host Doppler): 20 cm bins, 16 intervals, "
        "v_max 2 m/s, 1 Mbaud, 10 fps. Same path as the working linkbench 2D runs.",
        fft_mode=2, freq_start_mhz=58_000, range_max_mm=6400, range_res_mm=200, chirp_num=1,
        veloc_max_mm_s=2000, veloc_res_mm_s=250, dop_fft=False, frame_period_ms=100)
    P["doppler-2d-coarse"] = RadarConfig(
        name="doppler-2d-coarse", notes="2D, chip Doppler ON: 20 cm bins, 16 intervals, v_max 2 m/s, 1 Mbaud, 10 fps.",
        fft_mode=2, freq_start_mhz=58_000, range_max_mm=6400, range_res_mm=200, chirp_num=1,
        veloc_max_mm_s=2000, veloc_res_mm_s=250, dop_fft=True, frame_period_ms=100)
    P["doppler-2d-coarse-2mbaud"] = RadarConfig(
        name="doppler-2d-coarse-2mbaud", notes="Link test: as doppler-2d-coarse at 2 Mbaud and 20 fps "
        "(2 Mbaud never verified; 0 bytes on 2026-09-25 22:28 together with dop_fft 1).",
        fft_mode=2, freq_start_mhz=58_000, range_max_mm=6400, range_res_mm=200, chirp_num=1,
        veloc_max_mm_s=2000, veloc_res_mm_s=250, dop_fft=True, frame_period_ms=50, baudrate=2_000_000)
    P["doppler-2d-80mm"] = RadarConfig(
        name="doppler-2d-80mm", notes="2D, chip Doppler ON, 80 mm bins (N=64), 16 intervals, 1 Mbaud: 5 fps.",
        fft_mode=2, freq_start_mhz=58_000, range_max_mm=5120, range_res_mm=80, chirp_num=1,
        veloc_max_mm_s=2000, veloc_res_mm_s=250, dop_fft=True, frame_period_ms=200)
    # EXP-011 (owner: 1D with the finest resolution and a long max range looked best). Same 5.35 GHz sweep as
    # the 1D runs, 14.3 m max range = 512 samples (power of 2): +3 dB vs 7.7 m, IF high-pass corner at ~0.75 m (hand/handle/desk
    # attenuated ~5 dB more) and many empty bins beyond the room (correct noise estimate). 40 ms: the 10 ms
    # period of the owner's run lost 90 % of the chip frames on the host side (9.4 fps received).
    P["fine-1d-15m"] = RadarConfig(
        name="fine-1d-15m", notes="1D, 2.8 cm bins (B 5.35 GHz, N=512, no zero padding), 14.3 m max range, "
        "4 chirps, 25 fps. Needs link='spi' (6 KB cube). EXP-011.",
        fft_mode=1, freq_start_mhz=58_000, range_max_mm=14336, range_res_mm=28, chirp_num=4,
        frame_period_ms=40, link="spi")
    P["doppler-2d-40mm"] = RadarConfig(
        name="doppler-2d-40mm", notes="EXP-011: 2D with 2.5x finer range than doppler-2d-spi: 4 cm bins (B 3.75 GHz, "
        "N=128), 5.1 m, 16 intervals (v_res 25 cm/s), 24.6 KB cube, 25 fps. Needs link='spi'.",
        fft_mode=2, freq_start_mhz=58_000, range_max_mm=5120, range_res_mm=40, chirp_num=1,
        veloc_max_mm_s=2000, veloc_res_mm_s=250, dop_fft=True, frame_period_ms=40, link="spi")
    # EXP-013 "superset" recordings: finest range x most intervals that fit the link; coarser configs are then
    # emulated offline (emulate.py: fewer ADC samples at the same slope, fewer intervals, fewer frames).
    P["super-2d-30mm"] = RadarConfig(
        name="super-2d-30mm", notes="EXP-013 superset: 2D 3 cm bins (B 5.0 GHz, N=256), 7.68 m, 16 intervals "
        "(v_res 25 cm/s), 48 KB cube at 12.5 fps. Emulates 30-200 mm, K16/K8, 12.5/6 fps offline. SPI.",
        fft_mode=2, freq_start_mhz=58_000, range_max_mm=7680, range_res_mm=30, chirp_num=1,
        veloc_max_mm_s=2000, veloc_res_mm_s=250, dop_fft=True, frame_period_ms=80, link="spi")
    P["fine-2d-28mm-long"] = RadarConfig(
        name="fine-2d-28mm-long", notes="EXP-013: finest 2D the band allows (B 5.35 GHz only fits the slow slope of "
        "a long range): 2.8 cm bins, 14.3 m (IF high-pass ~0.72 m), 8 intervals (v_res 50 cm/s), 48 KB, 12.5 fps. SPI.",
        fft_mode=2, freq_start_mhz=58_000, range_max_mm=14336, range_res_mm=28, chirp_num=1,
        veloc_max_mm_s=2000, veloc_res_mm_s=500, dop_fft=True, frame_period_ms=80, link="spi")
    P["doppler-2d-spi"] = RadarConfig(
        name="doppler-2d-spi", notes="SPI-sized range-Doppler: 10 cm bins (N=64), 32 intervals, v_max 2 m/s "
        "(v_res 12.5 cm/s); 24.6 KB cube. Needs link='spi' (P1812).",
        fft_mode=2, freq_start_mhz=58_000, range_max_mm=6400, range_res_mm=100, chirp_num=1,
        veloc_max_mm_s=2000, veloc_res_mm_s=125, dop_fft=True, frame_period_ms=40, link="spi")
    # 2026-11-08 TCC collection (docs/protocols/TCC-001-collection.md). Both share the same range grid (40 mm,
    # 5.12 m, N=128, slope 293 MHz/us) so 1D and 2D can be compared bin for bin. The thesis' original config
    # (25 mm / 3.2 m) implies 468 MHz/us, above the 400 MHz/us limit (README D7).
    P["tcc-1d-40mm"] = RadarConfig(
        name="tcc-1d-40mm", notes="TCC: 1D, 4 cm bins (B 3.75 GHz, N=128), 5.12 m, 4 chirps, 30 ms (33 Hz). "
        "Fits the dual UART (1.5 KB/frame; linkbench UART 1d-128bins ran 40 fps with 0 loss), so the H563 board can "
        "run the same config. Recorded over SPI in the TCC session (same chip data).",
        fft_mode=1, freq_start_mhz=58_000, range_max_mm=5120, range_res_mm=40, chirp_num=4, frame_period_ms=30)
    P["tcc-2d-40mm"] = RadarConfig(
        name="tcc-2d-40mm", notes="TCC: 2D range-Doppler on the same range grid as tcc-1d-40mm: 4 cm, 5.12 m, "
        "16 intervals (v_max 2 m/s, v_res 0.25 m/s), chip Doppler, 50 ms (20 fps, below the 25 fps that lost frames). SPI.",
        fft_mode=2, freq_start_mhz=58_000, range_max_mm=5120, range_res_mm=40, chirp_num=1,
        veloc_max_mm_s=2000, veloc_res_mm_s=250, dop_fft=True, frame_period_ms=50, link="spi")
    # 2026-11-13: tcc-2d-40mm lost 26.5 % of its frames over SPI on the first test. Two lighter variants to compare on
    # the day; the recorder's "2D for steps" picks the one "Record step" uses.
    P["tcc-2d-40mm-12fps"] = RadarConfig(
        name="tcc-2d-40mm-12fps", notes="tcc-2d-40mm at 80 ms (12.5 fps): same cube, ~60 % of the link load. SPI.",
        fft_mode=2, freq_start_mhz=58_000, range_max_mm=5120, range_res_mm=40, chirp_num=1,
        veloc_max_mm_s=2000, veloc_res_mm_s=250, dop_fft=True, frame_period_ms=80, link="spi")
    P["tcc-2d-40mm-k8"] = RadarConfig(
        name="tcc-2d-40mm-k8", notes="tcc-2d-40mm with 8 Doppler intervals (v_res 0.5 m/s): half the cube, 20 fps. SPI.",
        fft_mode=2, freq_start_mhz=58_000, range_max_mm=5120, range_res_mm=40, chirp_num=1,
        veloc_max_mm_s=2000, veloc_res_mm_s=500, dop_fft=True, frame_period_ms=50, link="spi")
    try:                                   # private presets (product / regulatory): not in the public repo
        from .presets_private import private_presets
        P.update(private_presets())
    except ImportError:
        pass
    return P
