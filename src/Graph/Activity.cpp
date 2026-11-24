#include "Graph/Activity.h"

#include <algorithm>
#include <thread>

#ifdef MG_CUDA_SYNC
#include <cuda_runtime.h>
#endif

namespace mg
{
    void Activity::SetMode(Mode mode, int traceDelayMs)
    {
        m_traceDelayMs = std::clamp(traceDelayMs, 0, 5000);
        m_mode = mode;
    }

    void Activity::Reset()
    {
        std::lock_guard lock(m_mutex);
        m_nodes.clear();
        m_running.clear();
        m_forwardSeconds = 0;
        m_forwards = 0;
        m_synced = false;
    }

    Activity::Snapshot Activity::Get() const
    {
        Snapshot s;
        s.mode = m_mode;
        s.traceDelayMs = m_traceDelayMs;
        std::lock_guard lock(m_mutex);
        s.nodes = m_nodes;
        s.running = m_running;
        s.forwardSeconds = m_forwardSeconds;
        s.forwards = m_forwards;
        s.synced = m_synced;
        return s;
    }

    void Activity::Begin(const std::string &key)
    {
        std::lock_guard lock(m_mutex);
        m_running.push_back(key);
    }

    void Activity::End(const std::string &key, double seconds, bool nest)
    {
        std::lock_guard lock(m_mutex);
        Entry &e = m_nodes[key];
        e.seconds += seconds;
        ++e.calls;
        e.nest = nest;
        // Normally the last entry; search in case a node threw part-way.
        const auto it = std::find(m_running.rbegin(), m_running.rend(), key);
        if (it != m_running.rend()) m_running.erase(std::next(it).base());
    }

    void Activity::AddForward(double seconds)
    {
        std::lock_guard lock(m_mutex);
        m_forwardSeconds += seconds;
        ++m_forwards;
        m_running.clear();
    }

    void Activity::Sync(tensor::Device device)
    {
        if (device != tensor::Device::CUDA) return;
#ifdef MG_CUDA_SYNC
        cudaDeviceSynchronize();
        if (!m_synced)
        {
            std::lock_guard lock(m_mutex);
            m_synced = true;
        }
#endif
    }

    void Activity::Pause()
    {
        if (m_mode != Mode::Trace) return;
        using Clock = std::chrono::steady_clock;
        const auto start = Clock::now();
        // Short slices, so stopping or changing the delay takes effect at once.
        while (m_mode == Mode::Trace && !(m_stop && m_stop->load()))
        {
            const auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
            const int64_t left = m_traceDelayMs.load() - waited;
            if (left <= 0) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(std::min<int64_t>(left, 10)));
        }
        m_paused += std::chrono::duration<double>(Clock::now() - start).count();
    }

    const char *ModeName(Activity::Mode mode)
    {
        switch (mode)
        {
        case Activity::Mode::Heat: return "heat";
        case Activity::Mode::Trace: return "trace";
        default: return "off";
        }
    }

    Activity::Mode ParseMode(const std::string &name)
    {
        if (name == "heat") return Activity::Mode::Heat;
        if (name == "trace") return Activity::Mode::Trace;
        if (name == "off" || name.empty()) return Activity::Mode::Off;
        throw std::invalid_argument("unknown activity mode '" + name + "' (use off, heat or trace)");
    }
}
