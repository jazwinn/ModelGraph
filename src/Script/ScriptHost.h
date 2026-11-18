#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Graph/Registry.h"
#include "Script/PyNodes.h"

namespace mg::script
{
    struct ScriptStatus
    {
        std::filesystem::path file;
        std::string type;      // node type it registers (empty if it never loaded)
        bool ok = false;
        std::string error;     // last load error; the previous version stays live
        int reloads = 0;
    };

    // Watches folders of *.py files. Each file defines one node type; editing
    // a file re-registers it, deleting it unregisters it. Call Poll() regularly
    // (the editor does so every frame, throttled).
    class ScriptHost
    {
    public:
        explicit ScriptHost(Registry &registry) : m_registry(registry) {}

        void AddDirectory(const std::filesystem::path &dir);
        void ClearDirectories();
        const std::vector<std::filesystem::path> &Directories() const { return m_dirs; }

        // Returns true if any node type was added, changed or removed.
        bool Poll();

        std::vector<ScriptStatus> Statuses() const;

        // Writes a commented starter script and returns its path.
        static std::filesystem::path CreateStarter(const std::filesystem::path &dir, const std::string &name);

    private:
        struct Entry
        {
            std::filesystem::file_time_type mtime;
            std::shared_ptr<ScriptSlot> slot;
            ScriptStatus status;
        };

        bool Load(const std::filesystem::path &file, Entry &entry);

        Registry &m_registry;
        std::vector<std::filesystem::path> m_dirs;
        std::map<std::filesystem::path, Entry> m_entries;
    };
}
