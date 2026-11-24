#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include "Tensor/Tensor.h"

namespace mg
{
    // Records which nodes of a compiled model run and how long each takes, for
    // the editor's activity view. Nodes are keyed the way the canvas names
    // them: nest node ids joined by '/', then ':' and the node's own id
    // (":4" in the main graph, "3/7:12" inside two nests). Every copy of a
    // Repeat nest shares the keys of its subgraph, so a node's time is the sum
    // over all copies.
    //
    // One thread runs the model; any thread may read a Snapshot.
    class Activity
    {
    public:
        enum class Mode { Off, Heat, Trace };

        struct Entry
        {
            double seconds = 0;   // time spent in the node, pauses excluded
            uint64_t calls = 0;
            bool nest = false;    // includes the time of the nodes inside it
        };

        struct Snapshot
        {
            Mode mode = Mode::Off;
            int traceDelayMs = 0;
            std::map<std::string, Entry> nodes;
            std::vector<std::string> running;     // outermost first
            double forwardSeconds = 0;            // all forward passes, pauses excluded
            uint64_t forwards = 0;
            bool synced = false;                  // timings waited for the GPU
        };

        void SetMode(Mode mode, int traceDelayMs);
        Mode GetMode() const { return m_mode; }
        bool Enabled() const { return m_mode != Mode::Off; }

        // Pauses stop early once *stop becomes true.
        void SetStopFlag(const std::atomic<bool> *stop) { m_stop = stop; }

        void Reset();
        Snapshot Get() const;

        // Called by the runtime.
        void Begin(const std::string &key);
        void End(const std::string &key, double seconds, bool nest);
        void AddForward(double seconds);
        // Waits for queued GPU work so a node's time covers its kernels, not
        // just their launch. No-op on CPU.
        void Sync(tensor::Device device);
        // In trace mode, holds the current node on screen for the trace delay.
        void Pause();
        // Total time spent in Pause(); callers subtract it from wall time.
        double PausedSeconds() const { return m_paused; }

    private:
        mutable std::mutex m_mutex;
        std::map<std::string, Entry> m_nodes;
        std::vector<std::string> m_running;
        double m_forwardSeconds = 0;
        uint64_t m_forwards = 0;
        bool m_synced = false;

        std::atomic<Mode> m_mode{Mode::Off};
        std::atomic<int> m_traceDelayMs{0};
        const std::atomic<bool> *m_stop = nullptr;
        double m_paused = 0;   // only touched by the running thread
    };

    const char *ModeName(Activity::Mode mode);
    Activity::Mode ParseMode(const std::string &name);
}
