#pragma once

#include <filesystem>
#include <memory>
#include <mutex>
#include <string>

#include "Graph/Registry.h"

namespace mg::script
{
    struct ScriptState;

    // The live version of one script. Reloading swaps `current`; kernels built
    // from an older version pick up the new forward() on their next call.
    class ScriptSlot
    {
    public:
        std::shared_ptr<ScriptState> Current() const
        {
            std::lock_guard lock(m_mutex);
            return m_current;
        }
        void Set(std::shared_ptr<ScriptState> state)
        {
            std::lock_guard lock(m_mutex);
            m_current = std::move(state);
        }

    private:
        mutable std::mutex m_mutex;
        std::shared_ptr<ScriptState> m_current;
    };

    // Starts the embedded interpreter (idempotent, thread-safe).
    void EnsurePython();

    // Executes a .py file in a fresh namespace and reads its node definition.
    // Throws GraphError with the Python traceback on failure.
    std::shared_ptr<ScriptState> LoadScript(const std::filesystem::path &file);

    // The node type name a loaded script declares.
    std::string ScriptType(const ScriptState &state);

    // A NodeSpec whose infer/build/params call into the slot's current state.
    NodeSpec MakeSpec(const std::shared_ptr<ScriptSlot> &slot, const std::filesystem::path &file);
}
