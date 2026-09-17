#!/usr/bin/env python3
"""Localize GCS-link dropouts (GH #34: "Manual control lost" spam / Pilot Pro "No link").

Analyzes a tcpdump capture of the tablet<->BLACK TCP:5790 link and reports, in WALL time:

- inbound MANUAL_CONTROL (msg 69) inter-arrival gaps  -> what PX4 experiences as RC loss
- outbound (PX4 telemetry) stalls                     -> what the MCU display shows as "No link"
- per-gap discrimination via the sender's TCP timestamps (TSval): if the flushed burst's
  TSvals SPAN the gap the tablet kept writing and the network delayed delivery (WiFi/ARQ,
  expect retransmits); if they CLUSTER at flush time the tablet app itself went silent
  (Android throttling the backgrounded Mavlink Router App).

Capture (either topology; run on BLACK, then fly until the symptom shows):

    sudo tcpdump -i any -w /tmp/gcs5790.pcap 'tcp port 5790'
    uv run python tools/gcs_link_gaps.py /tmp/gcs5790.pcap

Live mode:  sudo tcpdump -i any -U -w - 'tcp port 5790' | uv run python tools/gcs_link_gaps.py -

Tracks each TCP flow separately, since the Ground Control Station (GCS) and the Mavlink Router App both
connect to :5790; the sticks flow is auto-detected as the inbound connection carrying MANUAL_CONTROL.
Stdlib-only; classic pcap (tcpdump default), EN10MB / LINUX_SLL / LINUX_SLL2, IPv4.
"""

from __future__ import annotations

import argparse
import struct
import sys
from dataclasses import dataclass, field

PORT = 5790
MANUAL_CONTROL = 69
MAV_V2, MAV_V1 = 0xFD, 0xFE


# --------------------------------------------------------------------------- pcap
def read_pcap_packets(f):
    """Yield (ts_seconds, linktype, raw_bytes) from a classic pcap stream."""
    hdr = f.read(24)
    if len(hdr) < 24:
        return
    magic = struct.unpack("<I", hdr[:4])[0]
    if magic in (0xA1B2C3D4, 0xA1B23C4D):
        endian, ns = "<", magic == 0xA1B23C4D
    elif magic in (0xD4C3B2A1, 0x4D3CB2A1):
        endian, ns = ">", magic == 0x4D3CB2A1
    else:
        sys.exit("not a classic pcap (pcapng? capture with tcpdump's default format)")
    linktype = struct.unpack(f"{endian}I", hdr[20:24])[0]
    while True:
        ph = f.read(16)
        if len(ph) < 16:
            return
        sec, frac, caplen, _wirelen = struct.unpack(f"{endian}IIII", ph)
        data = f.read(caplen)
        if len(data) < caplen:
            return
        yield sec + frac / (1e9 if ns else 1e6), linktype, data


def tcp_of(linktype, data):
    """Return (src_ip, dst_ip, src_port, dst_port, seq, tsval, payload) for IPv4/TCP."""
    if linktype == 1:  # EN10MB
        if len(data) < 14:
            return None
        et = struct.unpack(">H", data[12:14])[0]
        off = 14
        if et == 0x8100:
            et = struct.unpack(">H", data[16:18])[0]
            off = 18
        if et != 0x0800:
            return None
    elif linktype == 113:  # LINUX_SLL
        if len(data) < 16 or struct.unpack(">H", data[14:16])[0] != 0x0800:
            return None
        off = 16
    elif linktype == 276:  # LINUX_SLL2
        if len(data) < 20 or struct.unpack(">H", data[0:2])[0] != 0x0800:
            return None
        off = 20
    else:
        return None
    ip = data[off:]
    if len(ip) < 20 or ip[0] >> 4 != 4 or ip[9] != 6:
        return None
    ihl = (ip[0] & 0xF) * 4
    total = struct.unpack(">H", ip[2:4])[0]
    tcp = ip[ihl:total]
    if len(tcp) < 20:
        return None
    sport, dport = struct.unpack(">HH", tcp[0:4])
    seq = struct.unpack(">I", tcp[4:8])[0]
    doff = (tcp[12] >> 4) * 4
    tsval, i, opts = None, 0, tcp[20:doff]
    while i < len(opts):
        kind = opts[i]
        if kind == 0:
            break
        if kind == 1:
            i += 1
            continue
        if i + 1 >= len(opts) or opts[i + 1] < 2:
            break
        if kind == 8 and opts[i + 1] == 10:
            tsval = struct.unpack(">I", opts[i + 2 : i + 6])[0]
        i += opts[i + 1]
    sip = ".".join(str(b) for b in ip[12:16])
    dip = ".".join(str(b) for b in ip[16:20])
    return sip, dip, sport, dport, seq, tsval, tcp[doff:]


# ----------------------------------------------------------------- mavlink stream
@dataclass
class Direction:
    name: str
    next_seq: int | None = None
    pending: dict = field(default_factory=dict)  # seq -> (payload, ts, tsval)
    buf: bytearray = field(default_factory=bytearray)
    retransmits: int = 0
    holes: int = 0
    msgs: list = field(default_factory=list)  # (ts, tsval, sysid, compid, msgid)
    ts_pairs: list = field(default_factory=list)  # (ts, tsval) for tick estimate

    def add_segment(self, ts, seq, tsval, payload):
        if not payload:
            return
        if tsval is not None:
            self.ts_pairs.append((ts, tsval))
        if self.next_seq is None:
            self.next_seq = seq
        if (seq + len(payload) - self.next_seq) & 0xFFFFFFFF > 0x7FFFFFFF:
            self.retransmits += 1  # entirely at or below what the stream has consumed
            return
        self.pending[seq] = (payload, ts, tsval)
        self._drain()
        if sum(len(p) for p, _, _ in self.pending.values()) > 1 << 20:
            # unfilled hole, lost in capture: resync at the lowest buffered seq
            self.holes += 1
            self.next_seq = min(self.pending, key=lambda s: (s - self.next_seq) & 0xFFFFFFFF)
            self.buf.clear()
            self._drain()

    def _drain(self):
        while self.next_seq in self.pending:
            payload, ts, tsval = self.pending.pop(self.next_seq)
            self.next_seq = (self.next_seq + len(payload)) & 0xFFFFFFFF
            self.buf += payload
            self._parse(ts, tsval)

    def _parse(self, ts, tsval):
        b = self.buf
        i = 0
        while True:
            while i < len(b) and b[i] not in (MAV_V1, MAV_V2):
                i += 1
            if i + 3 > len(b):
                break
            if b[i] == MAV_V2:
                need = 12 + b[i + 1] + (13 if b[i + 2] & 0x01 else 0)
                if i + need > len(b):
                    break
                msgid = b[i + 7] | b[i + 8] << 8 | b[i + 9] << 16
                self.msgs.append((ts, tsval, b[i + 5], b[i + 6], msgid))
            else:
                need = 8 + b[i + 1]
                if i + need > len(b):
                    break
                self.msgs.append((ts, tsval, b[i + 3], b[i + 4], b[i + 5]))
            i += need
        del b[:i]


# ------------------------------------------------------------------------ analysis
def tick_ms(ts_pairs):
    """Estimate the sender's TSval tick in ms; Linux and Android typically run 1000 Hz."""
    if len(ts_pairs) < 10:
        return None
    (t0, v0), (t1, v1) = ts_pairs[0], ts_pairs[-1]
    if t1 - t0 < 5 or v1 == v0:
        return None
    return (t1 - t0) * 1e3 / ((v1 - v0) & 0xFFFFFFFF)


def gap_report(name, events, gap_s, tick, follow=0.25):
    """Print every inter-arrival gap > gap_s in `events` [(ts, tsval), ...]."""
    print(f"\n== {name}: gaps > {gap_s:.2f}s ==")
    n = 0
    t_start = events[0][0]
    for k in range(1, len(events)):
        gap = events[k][0] - events[k - 1][0]
        if gap <= gap_s:
            continue
        n += 1
        line = f"  t={events[k - 1][0] - t_start:8.2f}s  gap={gap:5.2f}s"
        burst = [tv for t, tv in events[k:] if t <= events[k][0] + follow and tv is not None]
        if tick and len(burst) >= 2:
            spread = (max(burst) - min(burst)) * tick / 1e3
            frac = spread / gap
            verdict = (
                "network-delayed (sender wrote through the gap)"
                if frac > 0.7
                else "SENDER-SILENT (app stalled)"
                if frac < 0.3
                else "mixed"
            )
            line += f"  sender-side spread of flushed burst={spread:.2f}s -> {verdict}"
        print(line)
        if n >= 60:
            print("  ... (truncated)")
            break
    if n == 0:
        print("  none")
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pcap", help="pcap file, or '-' for stdin (live tcpdump -U -w -)")
    ap.add_argument("--gap", type=float, default=0.35, help="report gaps above this (s)")
    args = ap.parse_args()

    # One Direction per TCP flow: the Ground Control Station (GCS) and the Mavlink Router App both
    # connect to :5790, so folding flows together would interleave unrelated seq spaces and give
    # garbage stats.
    conns: dict[tuple, Direction] = {}
    f = sys.stdin.buffer if args.pcap == "-" else open(args.pcap, "rb")
    for ts, linktype, raw in read_pcap_packets(f):
        r = tcp_of(linktype, raw)
        if r is None:
            continue
        sip, dip, sport, dport, seq, tsval, payload = r
        if PORT not in (sport, dport):
            continue
        key = (sip, sport, dip, dport)
        if key not in conns:
            name = f"{sip}:{sport} -> :{PORT}" if dport == PORT else f":{PORT} -> {dip}:{dport}"
            conns[key] = Direction(name)
        conns[key].add_segment(ts, seq, tsval, payload)

    flows = [d for d in conns.values() if d.msgs]
    if not flows:
        sys.exit("no MAVLink on tcp:5790 in this capture")

    for d in flows:
        span = max(d.msgs[-1][0] - d.msgs[0][0], 1e-9)
        by_src = {}
        for _, _, sysid, compid, msgid in d.msgs:
            by_src[(sysid, compid, msgid)] = by_src.get((sysid, compid, msgid), 0) + 1
        top = sorted(by_src.items(), key=lambda kv: -kv[1])[:6]
        print(
            f"\n== {d.name}: {len(d.msgs)} msgs over {span:.0f}s, "
            f"{d.retransmits} retransmitted segs, {d.holes} capture holes =="
        )
        for (sysid, compid, msgid), cnt in top:
            print(f"  sys {sysid:3d} comp {compid:3d} msg {msgid:5d}: {cnt:6d} ({cnt / span:5.1f}/s)")

    # the sticks flow is whichever inbound connection carries MANUAL_CONTROL
    mc_flow = next((d for d in flows if any(m == MANUAL_CONTROL for *_x, m in d.msgs)), None)
    if mc_flow:
        tick_in = tick_ms(mc_flow.ts_pairs)
        if tick_in:
            print(
                f"\nsticks-flow sender TSval tick ~ {tick_in:.2f} ms "
                f"({'plausible' if 0.5 < tick_in < 20 else 'IMPLAUSIBLE - ignore verdicts'})"
            )
        mc = [(t, tv) for t, tv, _s, _c, m in mc_flow.msgs if m == MANUAL_CONTROL]
        print(f"\nMANUAL_CONTROL: {len(mc)} msgs from the MCU (rate {len(mc) / (mc[-1][0] - mc[0][0]):.1f}/s)")
        n = gap_report(f"inbound MANUAL_CONTROL ({mc_flow.name})", mc, args.gap, tick_in)
        print(
            f"  {n} gaps > {args.gap}s; PX4 declares manual-control loss after COM_RC_LOSS_T "
            "(0.5s ff_base default; 1.0s once the Doodle input layer has applied)"
        )
    else:
        print("\nno MANUAL_CONTROL inbound (sticks idle or MCU path down)")

    for d in flows:
        if d.name.startswith(f":{PORT}"):
            gap_report(
                f"outbound telemetry {d.name} (the MCU 'No link' side rides the router-app flow)",
                [(t, tv) for t, tv, *_ in d.msgs],
                args.gap,
                tick_ms(d.ts_pairs),
            )


if __name__ == "__main__":
    main()
