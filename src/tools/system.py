"""System volume control (spec §8.2, §8.3 tool 1).

pycaw's API moved: `AudioUtilities.GetSpeakers()` returns an `AudioDevice` that
no longer has `.Activate()`, so the widely-copied
`devices.Activate(IAudioEndpointVolume._iid_, ...)` snippet raises
AttributeError on pycaw 20251023. The device now exposes `.EndpointVolume`
directly, which is what this uses.

COM needs initialising on whichever thread touches it, and tools run on the turn
worker rather than the main thread, so every call does that itself.
"""

from __future__ import annotations

from .registry import Tool, ToolError, ToolResult


def _endpoint_volume():
    import comtypes
    from pycaw.pycaw import AudioUtilities

    try:
        comtypes.CoInitialize()
    except Exception:
        pass  # already initialised on this thread
    device = AudioUtilities.GetSpeakers()
    endpoint = getattr(device, "EndpointVolume", None)
    if endpoint is None:
        raise ToolError("could not reach the audio endpoint")
    return device, endpoint


def _describe(level: int, muted: bool) -> str:
    if muted:
        return f"Muted. Volume is at {level} percent."
    return f"Volume is at {level} percent."


def system_volume(action: str, level: int | None = None) -> ToolResult:
    device, endpoint = _endpoint_volume()
    current = int(round(endpoint.GetMasterVolumeLevelScalar() * 100))
    muted = bool(endpoint.GetMute())

    if action == "get":
        return ToolResult(True, _describe(current, muted),
                          {"level": current, "muted": muted, "device": device.FriendlyName})

    if action == "set":
        if level is None:
            raise ToolError("set needs a level between 0 and 100")
        level = max(0, min(100, int(level)))
        endpoint.SetMasterVolumeLevelScalar(level / 100.0, None)
        if muted and level > 0:
            # Setting a level while muted would otherwise change nothing audible.
            endpoint.SetMute(0, None)
        return ToolResult(True, f"Volume set to {level} percent.", {"level": level})

    if action in ("up", "down"):
        step = 10 if level is None else max(1, min(100, int(level)))
        target = max(0, min(100, current + (step if action == "up" else -step)))
        endpoint.SetMasterVolumeLevelScalar(target / 100.0, None)
        return ToolResult(True, f"Volume {action} to {target} percent.", {"level": target})

    if action == "mute":
        endpoint.SetMute(1, None)
        return ToolResult(True, "Muted.", {"muted": True})

    if action == "unmute":
        endpoint.SetMute(0, None)
        return ToolResult(True, f"Unmuted, at {current} percent.", {"muted": False})

    raise ToolError(f"unknown action {action!r}")


# ---------------------------------------------------------------------------
# system_info: where am I, what time is it, how is the machine doing
# ---------------------------------------------------------------------------


_TUNNEL_HINTS = ("wireguard", "vpn", "tap-", "openvpn", "tunnel", "wintun")

# Mesh/remote-access tools that create a tunnel adapter but do NOT carry general
# internet traffic. Listing them by name would be a losing game, which is why
# the check below looks at routing instead.
_NOT_A_VPN_FOR_TRAFFIC = ("tailscale", "zerotier", "hamachi", "radmin")


def _vpn_on_default_route() -> str:
    """The tunnel adapter carrying internet traffic, or "" if there is none.

    Naively scanning adapter names for "tunnel" reports a false positive on
    anything that creates a virtual interface. Tailscale was up on this machine
    with no VPN running, and the location tool kept insisting a VPN was active
    after the user had turned ProtonVPN off.

    What actually matters for IP geolocation is which adapter carries the
    default route, because that is the one that decides the exit address.
    """
    import subprocess

    script = (
        "$r = Get-NetRoute -DestinationPrefix '0.0.0.0/0' -ErrorAction SilentlyContinue | "
        "Sort-Object RouteMetric, ifMetric | Select-Object -First 1; "
        "if ($r) { (Get-NetAdapter -ifIndex $r.ifIndex -ErrorAction SilentlyContinue)"
        ".InterfaceDescription }"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.SubprocessError):
        return ""

    adapter = (out.stdout or "").strip()
    if not adapter:
        return ""
    lowered = adapter.lower()
    if any(name in lowered for name in _NOT_A_VPN_FOR_TRAFFIC):
        return ""
    if any(hint in lowered for hint in _TUNNEL_HINTS):
        return adapter
    return ""


def _ip_location() -> dict:
    import httpx

    response = httpx.get(
        "http://ip-api.com/json/",
        params={
            "fields": "status,message,country,regionName,city,lat,lon,timezone,"
                      "offset,isp,query"
        },
        timeout=8.0,
        headers={"User-Agent": "jarvis/1.0"},
    )
    response.raise_for_status()
    data = response.json()
    if data.get("status") != "success":
        raise ToolError(data.get("message", "the location service refused"))
    return data


def _local_utc_offset_seconds() -> int:
    """This machine's current offset from UTC, in seconds."""
    import datetime

    offset = datetime.datetime.now().astimezone().utcoffset()
    return int(offset.total_seconds()) if offset else 0


def system_info(action: str) -> ToolResult:
    import datetime

    if action == "time":
        now = datetime.datetime.now()
        return ToolResult(True, f"It's {now.strftime('%I:%M %p').lstrip('0')}.",
                          {"iso": now.isoformat()})

    if action == "date":
        now = datetime.datetime.now()
        return ToolResult(True, f"It's {now.strftime('%A, %B %d, %Y')}.",
                          {"iso": now.date().isoformat()})

    if action == "location":
        try:
            data = _ip_location()
        except ToolError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ToolError(f"couldn't reach the location service ({type(exc).__name__})") from exc

        city = data.get("city") or "somewhere"
        region = data.get("regionName") or ""
        country = data.get("country") or ""
        where = ", ".join(p for p in (city, region, country) if p)

        vpn = _vpn_on_default_route()
        ip_tz = str(data.get("timezone") or "")

        # Compare UTC OFFSETS, not timezone names. Matching "Europe/Berlin"
        # against Windows' "W. Europe Standard Time" by string never works, and
        # the earlier attempt reported a mismatch for a machine whose clock
        # agreed with its IP perfectly.
        ip_offset = data.get("offset")
        offset_mismatch = False
        if isinstance(ip_offset, (int, float)):
            offset_mismatch = abs(int(ip_offset) - _local_utc_offset_seconds()) > 3600

        # An IP location is only ever a guess about the network, not the person.
        # Say so when the evidence disagrees rather than sounding certain.
        if vpn:
            summary = (
                f"Your internet connection comes out in {where}, but you have a VPN "
                f"running, so that's probably where the VPN is, not where you are."
            )
        elif offset_mismatch:
            summary = (
                f"Your connection looks like it's in {where}, though your computer's "
                f"clock is set to a different time zone, so I'm not certain."
            )
        else:
            summary = f"Based on your internet connection, you're in {where}."

        return ToolResult(
            True, summary,
            {"city": city, "region": region, "country": country,
             "lat": data.get("lat"), "lon": data.get("lon"),
             "ip_timezone": ip_tz, "vpn_active": bool(vpn), "vpn_adapter": vpn,
             "isp": data.get("isp")},
        )

    if action == "battery":
        import psutil

        battery = psutil.sensors_battery()
        if battery is None:
            return ToolResult(True, "This machine doesn't have a battery.", {"battery": None})
        plugged = " and plugged in" if battery.power_plugged else ""
        return ToolResult(True, f"Battery is at {int(battery.percent)} percent{plugged}.",
                          {"percent": battery.percent, "plugged": battery.power_plugged})

    if action == "performance":
        import shutil

        import psutil

        cpu = psutil.cpu_percent(interval=0.3)
        mem = psutil.virtual_memory()
        disk = shutil.disk_usage("C:\\")
        return ToolResult(
            True,
            f"CPU is at {cpu:.0f} percent, memory at {mem.percent:.0f} percent, "
            f"and the C drive has {disk.free / 1e9:.0f} gigabytes free.",
            {"cpu": cpu, "memory_percent": mem.percent, "disk_free_gb": disk.free / 1e9},
        )

    raise ToolError(f"unknown action {action!r}")


INFO_TOOL = Tool(
    name="system_info",
    description=(
        "Look up the current time or date, roughly where the user is, battery "
        "level, or how busy the computer is."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["time", "date", "location", "battery", "performance"],
                "description": "What to look up.",
            }
        },
        "required": ["action"],
    },
    handler=system_info,
)


# ---------------------------------------------------------------------------
# screenshot_describe: §8.3 tool 9, explicitly a stub until a VLM exists
# ---------------------------------------------------------------------------


def screenshot_describe(question: str | None = None) -> ToolResult:
    return ToolResult(
        False,
        "I can't see the screen yet. That needs a vision model, which isn't set up.",
        {"stub": True, "question": question},
    )


SCREENSHOT_TOOL = Tool(
    name="screenshot_describe",
    description=(
        "Look at what is on screen and describe it. NOT AVAILABLE YET: this "
        "always reports that it cannot see, so prefer window_info for questions "
        "about which programs are open."
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "What to look for on screen."}
        },
        "required": [],
    },
    handler=screenshot_describe,
)


TOOL = Tool(
    name="system_volume",
    description=(
        "Control the computer's speaker volume. Use for anything about making it "
        "louder, quieter, muting, or checking the current level."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["get", "set", "up", "down", "mute", "unmute"],
                "description": "What to do with the volume.",
            },
            "level": {
                "type": "integer",
                "description": (
                    "For 'set', the target volume 0-100. For 'up'/'down', how many "
                    "percent to change by. Omit for other actions."
                ),
            },
        },
        "required": ["action"],
    },
    handler=system_volume,
)
