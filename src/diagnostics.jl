# Runtime diagnostics for long-running QAOA computations.
#
# Controlled by environment variables (no code changes needed):
#   QAOA_DIAG=1          enable all diagnostics (default: off)
#   QAOA_DIAG_MEM=1      enable memory reporting
#   QAOA_DIAG_TIMING=1   enable per-phase timing
#   QAOA_DIAG_PROGRESS=1 enable iteration progress
#   QAOA_DIAG_LEVEL=2    verbosity: 0=off, 1=summary, 2=detail, 3=trace
#   QAOA_DIAG_FILE=path  also write diagnostics to this file (append)
#
# Usage from code:
#   diag_mem("label")           — log RSS if memory diagnostics enabled
#   diag_time("label", t)       — log elapsed time if timing enabled
#   diag_progress("label", ...) — log progress if progress enabled
#   diag_phase("label") do ... end — time a block and log it

module Diagnostics

# No external dependencies — uses only Base and stdlib that's always available

# ── Configuration (re-read at __init__ to respect runtime env) ────

const _DIAG_MEM      = Ref(false)
const _DIAG_TIMING   = Ref(false)
const _DIAG_PROGRESS = Ref(false)
const _DIAG_LEVEL    = Ref(0)

const _diag_io = Ref{IO}(stderr)
const _start_time = Ref(time_ns())
const _peak_rss = Ref(0.0)

function __init__()
    _start_time[] = time_ns()

    all     = get(ENV, "QAOA_DIAG", "0") == "1"
    _DIAG_MEM[]      = all || get(ENV, "QAOA_DIAG_MEM", "0") == "1"
    _DIAG_TIMING[]   = all || get(ENV, "QAOA_DIAG_TIMING", "0") == "1"
    _DIAG_PROGRESS[] = all || get(ENV, "QAOA_DIAG_PROGRESS", "0") == "1"
    _DIAG_LEVEL[]    = parse(Int, get(ENV, "QAOA_DIAG_LEVEL", all ? "2" : "0"))

    diag_file = get(ENV, "QAOA_DIAG_FILE", "")
    if !isempty(diag_file)
        try
            _diag_io[] = open(diag_file, "a")
        catch e
            @warn "QAOA_DIAG_FILE=$(diag_file) could not be opened: $e"
        end
    end

    # Register atexit hook unconditionally — this is cheap and catches OOM kills
    atexit() do
        _emit("[EXIT] $(_timestamp()) " *
              "RSS=$(round(rss_gb(), digits=1))GB " *
              "peak=$(round(_peak_rss[], digits=1))GB " *
              "elapsed=$(round(session_elapsed(), digits=0))s")
        if _diag_io[] !== stderr && _diag_io[] isa IOStream
            close(_diag_io[])
        end
    end

    if _DIAG_LEVEL[] > 0
        _emit("[INIT] PID=$(getpid()) Julia=$(VERSION) RAM=$(round(Sys.total_memory()/(1024^3), digits=1))GB level=$(_DIAG_LEVEL[])")
    end
end

# ── Internal helpers ─────────────────────────────────────────────

function _timestamp()
    t = time()
    secs = floor(Int, t)
    lt = Libc.TmStruct(secs)
    h = lpad(lt.hour, 2, '0')
    m = lpad(lt.min, 2, '0')
    s = lpad(lt.sec, 2, '0')
    "$h:$m:$s"
end

function _emit(msg::String)
    elapsed = round(Int, session_elapsed())
    ts = "[$(_timestamp()) +$(lpad(elapsed, 6))s]"
    line = "$ts $msg"
    try
        println(stderr, line)
        flush(stderr)
    catch; end
    if _diag_io[] !== stderr
        try
            println(_diag_io[], line)
            flush(_diag_io[])
        catch; end
    end
end

function session_elapsed()
    (time_ns() - _start_time[]) / 1e9
end

function rss_gb()
    try
        pid = getpid()
        rss_kb = parse(Int, strip(read(`ps -p $pid -o rss=`, String)))
        gb = rss_kb / (1024^2)
        _peak_rss[] = max(_peak_rss[], gb)
        return gb
    catch
        return Sys.maxrss() / 1e9
    end
end

# ── Public API ───────────────────────────────────────────────────

"""
    diag_mem(label)

Log current RSS if memory diagnostics are enabled (QAOA_DIAG_MEM=1 or QAOA_DIAG=1).
"""
function diag_mem(label::String)
    _DIAG_MEM[] || return
    gb = rss_gb()
    _emit("[MEM] $label: RSS=$(round(gb, digits=1))GB peak=$(round(_peak_rss[], digits=1))GB total=$(round(Sys.total_memory()/(1024^3), digits=0))GB")
end

"""
    diag_time(label, seconds; level=1)

Log an elapsed time if timing diagnostics are enabled.
"""
function diag_time(label::String, seconds::Real; level::Int=1)
    (_DIAG_TIMING[] && _DIAG_LEVEL[] >= level) || return
    _emit("[TIME] $label: $(round(seconds, digits=2))s")
end

"""
    diag_progress(label, args...)

Log progress info (iterations, values, etc.) if progress diagnostics are enabled.
"""
function diag_progress(label::String, msg::String)
    _DIAG_PROGRESS[] || return
    _emit("[PROG] $label: $msg")
end

"""
    diag_phase(label, f)

Time a block and log it. Returns the block's result.
Usage: `result = diag_phase("forward p=14") do ... end`
"""
function diag_phase(f::Function, label::String)
    if _DIAG_TIMING[] && _DIAG_LEVEL[] >= 2
        diag_mem("before $label")
    end
    t = @elapsed result = f()
    diag_time(label, t)
    if _DIAG_MEM[] && _DIAG_LEVEL[] >= 2
        diag_mem("after $label")
    end
    result
end

"""
    diag_info(msg; level=1)

Log an informational message at the given verbosity level.
"""
function diag_info(msg::String; level::Int=1)
    _DIAG_LEVEL[] >= level || return
    _emit("[INFO] $msg")
end

"""
    diag_warn(msg)

Always log a warning (regardless of diagnostic level).
"""
function diag_warn(msg::String)
    _emit("[WARN] $msg")
end

"""
    enabled()

Returns true if any diagnostics are enabled.
"""
enabled() = _DIAG_LEVEL[] > 0

end # module Diagnostics
