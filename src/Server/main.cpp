// mg_server: serves the ModelGraph web editor and a JSON API over the graph
// compiler, the InferenceEngine runtime and the Python node scripts.
//
//   mg_server [--port 8710] [--root <project dir>] [--no-browser]
//
// Listens on 127.0.0.1 only: the API can run Python and write files under
// the project's graphs/ and scripts/ folders.

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <regex>
#include <sstream>
#include <thread>

#include <httplib.h>
#include <nlohmann/json.hpp>

#include "Graph/Analyze.h"
#include "Graph/Templates.h"
#include "Script/ScriptHost.h"
#include "Session/Session.h"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#include <commdlg.h>
#include <objbase.h>
#endif

namespace fs = std::filesystem;
using nlohmann::json;

namespace
{
    // ---- JSON views of the graph layer -----------------------------------------

    const char *KindName(mg::AttrKind k)
    {
        switch (k)
        {
        case mg::AttrKind::Bool: return "bool";
        case mg::AttrKind::Int: return "int";
        case mg::AttrKind::Float: return "float";
        case mg::AttrKind::String: return "string";
        case mg::AttrKind::Enum: return "enum";
        case mg::AttrKind::Shape: return "shape";
        case mg::AttrKind::Subgraph: return "subgraph";
        case mg::AttrKind::NodeRef: return "noderef";
        }
        return "string";
    }

    const char *RoleName(mg::NodeRole r)
    {
        return r == mg::NodeRole::Input ? "input" : r == mg::NodeRole::Output ? "output" : "op";
    }

    json PortJson(const mg::PortSpec &p)
    {
        return {{"name", p.name}, {"dtype", p.dtype ? json(mg::DTypeName(*p.dtype)) : json(nullptr)}, {"optional", p.optional}};
    }

    json PortsJson(const std::vector<mg::PortSpec> &ports)
    {
        json a = json::array();
        for (const auto &p : ports) a.push_back(PortJson(p));
        return a;
    }

    json SpecJson(const mg::NodeSpec &s)
    {
        json attrs = json::array();
        for (const auto &a : s.attrs)
        {
            attrs.push_back({{"name", a.name}, {"kind", KindName(a.kind)}, {"default", mg::AttrToJson(a.def)},
                             {"options", a.options}, {"help", a.help}});
        }
        return {{"type", s.type}, {"category", s.category}, {"description", s.description}, {"source", s.source},
                {"role", RoleName(s.role)}, {"inputs", PortsJson(s.inputs)}, {"outputs", PortsJson(s.outputs)},
                {"attrs", attrs}, {"dynamicPorts", static_cast<bool>(s.dynamicPorts)}};
    }

    json TypeJson(const mg::TensorType &t)
    {
        json shape = json::array();
        for (const auto &d : t.shape) shape.push_back(d.IsKnown() ? json(d.value) : json(d.Text()));
        return {{"dtype", mg::DTypeName(t.dtype)}, {"shape", shape}, {"text", t.Text()}};
    }

    json AnalysisJson(const mg::Analysis &a)
    {
        json nodes = json::object();
        for (const auto &[id, ports] : a.ports)
        {
            json n = {{"ports", {{"inputs", PortsJson(ports.inputs)}, {"outputs", PortsJson(ports.outputs)}}}};
            if (auto it = a.inputs.find(id); it != a.inputs.end())
            {
                json ins = json::array();
                for (const auto &t : it->second) ins.push_back(t ? TypeJson(*t) : json(nullptr));
                n["inputs"] = ins;
            }
            if (auto it = a.outputs.find(id); it != a.outputs.end())
            {
                json outs = json::array();
                for (const auto &t : it->second) outs.push_back(TypeJson(t));
                n["outputs"] = outs;
            }
            if (auto it = a.params.find(id); it != a.params.end()) n["params"] = it->second;
            nodes[std::to_string(id)] = n;
        }
        json errors = json::object();
        for (const auto &[id, e] : a.errors) errors[std::to_string(id)] = e;
        return {{"nodes", nodes}, {"errors", errors}, {"graphErrors", a.graphErrors}, {"totalParams", a.totalParams}};
    }

    // ---- server state ----------------------------------------------------------

    struct App
    {
        fs::path root, graphs, scripts, web;
        mg::Registry registry;
        std::unique_ptr<mg::script::ScriptHost> host;
        std::mutex hostMutex;
        std::unique_ptr<mg::Session> session;
        std::atomic<bool> running{true};
        std::atomic<uint64_t> scriptGeneration{0};
    };

    // File names from the client: a plain name inside one of our folders.
    fs::path SafePath(const fs::path &dir, const std::string &name, const char *extension)
    {
        static const std::regex ok(R"(^[A-Za-z0-9_][A-Za-z0-9_. \-]*$)");
        if (!std::regex_match(name, ok) || name.find("..") != std::string::npos || !name.ends_with(extension))
        {
            throw mg::GraphError("invalid file name '" + name + "'");
        }
        return dir / name;
    }

    std::string ReadText(const fs::path &p)
    {
        std::ifstream in(p, std::ios::binary);
        if (!in) throw mg::GraphError("cannot read " + p.filename().string());
        std::stringstream s;
        s << in.rdbuf();
        return s.str();
    }

    void WriteText(const fs::path &p, const std::string &text)
    {
        fs::create_directories(p.parent_path());
        std::ofstream out(p, std::ios::binary);
        if (!out) throw mg::GraphError("cannot write " + p.filename().string());
        out << text;
    }

    // Paths travel as UTF-8 strings in JSON.
    fs::path FromUtf8(const std::string &s)
    {
        return fs::path(std::u8string(s.begin(), s.end()));
    }

    std::string ToUtf8(const fs::path &p)
    {
        const std::u8string u = p.u8string();
        return std::string(u.begin(), u.end());
    }

    // A document is identified by a name inside graphs/ ("gpt2.json") or, when
    // opened through the file dialog, by an absolute path.
    fs::path GraphPath(const App &app, const std::string &file)
    {
        if (file.empty()) return app.graphs / "untitled.json";
        const fs::path p = FromUtf8(file);
        if (p.is_absolute())
        {
            if (p.extension() != ".json") throw mg::GraphError("graphs are .json files");
            return p.lexically_normal();
        }
        return SafePath(app.graphs, file, ".json");
    }

    mg::Document DocFrom(const App &app, const json &body)
    {
        mg::Document doc = mg::Document::FromJson(body.at("doc"));
        doc.path = GraphPath(app, body.value("file", ""));
        return doc;
    }

#ifdef _WIN32
    // Native open/save dialogs, raised in front of the browser that asked.
    std::optional<fs::path> FileDialog(const fs::path &initialDir, const std::string &suggested, bool save)
    {
        CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

        // A background process cannot normally take focus; borrow the
        // foreground window's input queue so the dialog appears on top.
        HWND owner = CreateWindowExW(WS_EX_TOPMOST | WS_EX_TOOLWINDOW, L"STATIC", L"ModelGraph", WS_POPUP, 0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr);
        const DWORD fgThread = GetWindowThreadProcessId(GetForegroundWindow(), nullptr);
        const DWORD me = GetCurrentThreadId();
        AttachThreadInput(fgThread, me, TRUE);
        SetForegroundWindow(owner);
        AttachThreadInput(fgThread, me, FALSE);

        wchar_t buffer[MAX_PATH * 4] = {};
        const std::wstring name = FromUtf8(suggested).wstring();
        wcsncpy_s(buffer, name.c_str(), _TRUNCATE);
        const std::wstring dir = initialDir.wstring();

        OPENFILENAMEW ofn = {};
        ofn.lStructSize = sizeof(ofn);
        ofn.hwndOwner = owner;
        ofn.lpstrFilter = L"ModelGraph graphs (*.json)\0*.json\0All files\0*.*\0";
        ofn.lpstrFile = buffer;
        ofn.nMaxFile = static_cast<DWORD>(std::size(buffer));
        ofn.lpstrInitialDir = dir.c_str();
        ofn.lpstrDefExt = L"json";
        ofn.Flags = OFN_NOCHANGEDIR | OFN_EXPLORER | (save ? OFN_OVERWRITEPROMPT : OFN_FILEMUSTEXIST);
        const BOOL ok = save ? GetSaveFileNameW(&ofn) : GetOpenFileNameW(&ofn);
        DestroyWindow(owner);
        if (!ok) return std::nullopt;
        return fs::path(buffer);
    }
#else
    std::optional<fs::path> FileDialog(const fs::path &, const std::string &, bool) { return std::nullopt; }
#endif

    // Graphs inside graphs/ keep their short name; anything else its full path.
    std::string FileId(const App &app, const fs::path &p)
    {
        std::error_code ec;
        if (fs::equivalent(p.parent_path(), app.graphs, ec)) return ToUtf8(p.filename());
        return ToUtf8(p);
    }

    void Reply(httplib::Response &res, const json &j, int status = 200)
    {
        res.status = status;
        res.set_content(j.dump(), "application/json");
    }

    template <typename F>
    httplib::Server::Handler Api(F f)
    {
        return [f](const httplib::Request &req, httplib::Response &res)
        {
            try
            {
                const json body = req.body.empty() ? json::object() : json::parse(req.body);
                Reply(res, f(req, body));
            }
            catch (const std::exception &e)
            {
                Reply(res, {{"error", e.what()}}, 400);
            }
        };
    }

    fs::path FindRoot(const char *argv0)
    {
        // The project root holds graphs/ and web/; look upwards from the exe and cwd.
        for (fs::path start : {fs::current_path(), fs::absolute(argv0).parent_path()})
        {
            for (fs::path p = start; !p.empty(); p = p.parent_path())
            {
                if (fs::is_directory(p / "web") && fs::is_directory(p / "graphs")) return p;
                if (p == p.parent_path()) break;
            }
        }
        return fs::current_path();
    }

    void OpenBrowser(const std::string &url)
    {
#ifdef _WIN32
        ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
        std::system(("xdg-open " + url + " >/dev/null 2>&1 &").c_str());
#endif
    }
}

int main(int argc, char **argv)
{
    int port = 8710;
    bool browser = true;
    App app;
    app.root = FindRoot(argv[0]);
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--port" && i + 1 < argc) port = std::stoi(argv[++i]);
        else if (a == "--root" && i + 1 < argc) app.root = fs::absolute(argv[++i]);
        else if (a == "--no-browser") browser = false;
    }
    app.graphs = app.root / "graphs";
    app.scripts = app.root / "scripts";
    app.web = app.root / "web" / "dist";
    fs::create_directories(app.graphs);
    fs::create_directories(app.scripts);

    mg::RegisterBuiltins(app.registry);
    app.host = std::make_unique<mg::script::ScriptHost>(app.registry);
    app.host->AddDirectory(app.scripts);
    app.session = std::make_unique<mg::Session>(app.registry);

    // Hot reload: rescan scripts/ a few times a second.
    std::thread watcher([&app]
    {
        while (app.running)
        {
            {
                std::lock_guard lock(app.hostMutex);
                if (app.host->Poll()) ++app.scriptGeneration;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
        }
    });

    httplib::Server svr;

    svr.Get("/api/info", Api([&](const auto &, const json &) -> json
    {
#ifdef IE_ENABLE_CUDA
        const bool cuda = true;
#else
        const bool cuda = false;
#endif
        return {{"root", app.root.string()}, {"graphs", app.graphs.string()}, {"scripts", app.scripts.string()}, {"cuda", cuda}};
    }));

    svr.Get("/api/nodes", Api([&](const auto &, const json &) -> json
    {
        json specs = json::array();
        for (const auto &s : app.registry.All()) specs.push_back(SpecJson(*s));
        return {{"version", app.registry.Version()}, {"specs", specs}};
    }));

    svr.Post("/api/analyze", Api([&](const auto &, const json &body) -> json
    {
        const mg::Document doc = DocFrom(app, body);
        json graphs = json::object();
        graphs["main"] = AnalysisJson(mg::Analyze(doc, doc.main, app.registry));
        for (const auto &[name, g] : doc.subgraphs) graphs[name] = AnalysisJson(mg::Analyze(doc, g, app.registry));
        json vars = json::object();
        for (const auto &[k, v] : doc.NumericVariables()) vars[k] = v;
        return {{"graphs", graphs}, {"variables", vars}, {"registryVersion", app.registry.Version()}};
    }));

    // ---- graphs & templates ----
    svr.Get("/api/graphs", Api([&](const auto &, const json &) -> json
    {
        json files = json::array();
        for (const auto &e : fs::directory_iterator(app.graphs))
        {
            if (e.is_regular_file() && e.path().extension() == ".json") files.push_back(e.path().filename().string());
        }
        return {{"files", files}};
    }));
    svr.Get(R"(/api/graphs/(.+))", Api([&](const httplib::Request &req, const json &) -> json
    {
        return json::parse(ReadText(SafePath(app.graphs, req.matches[1], ".json")));
    }));
    svr.Put(R"(/api/graphs/(.+))", Api([&](const httplib::Request &req, const json &body) -> json
    {
        const fs::path p = SafePath(app.graphs, req.matches[1], ".json");
        mg::Document::FromJson(body);   // validate before writing
        WriteText(p, body.dump(2) + "\n");
        return {{"saved", p.filename().string()}};
    }));
    // Any graph file by absolute path (from the dialog or the recent list).
    svr.Get("/api/file", Api([&](const httplib::Request &req, const json &) -> json
    {
        return json::parse(ReadText(GraphPath(app, req.get_param_value("path"))));
    }));
    svr.Put("/api/file", Api([&](const httplib::Request &req, const json &body) -> json
    {
        const fs::path p = GraphPath(app, req.get_param_value("path"));
        mg::Document::FromJson(body);
        WriteText(p, body.dump(2) + "\n");
        return {{"saved", FileId(app, p)}};
    }));
    svr.Post("/api/dialog/open", Api([&](const auto &, const json &) -> json
    {
        const auto p = FileDialog(app.graphs, "", false);
        if (!p) return {{"cancelled", true}};
        return {{"file", FileId(app, *p)}, {"doc", json::parse(ReadText(*p))}};
    }));
    svr.Post("/api/dialog/save", Api([&](const auto &, const json &body) -> json
    {
        const auto p = FileDialog(app.graphs, body.value("suggested", "model.json"), true);
        if (!p) return {{"cancelled", true}};
        return {{"file", FileId(app, p->extension() == ".json" ? *p : fs::path(*p) += ".json")}};
    }));

    svr.Get("/api/templates", Api([&](const auto &, const json &) -> json
    {
        json list = json::array();
        for (const auto &t : mg::Templates()) list.push_back({{"id", t.id}, {"title", t.title}});
        return {{"templates", list}};
    }));
    svr.Get(R"(/api/templates/(.+))", Api([&](const httplib::Request &req, const json &) -> json
    {
        return mg::MakeTemplate(req.matches[1]).ToJson();
    }));

    // ---- run ----
    svr.Post("/api/build", Api([&](const auto &, const json &body) -> json
    {
        const std::string mode = body.value("mode", "fresh");
        const mg::BuildMode m = mode == "checkpoint" ? mg::BuildMode::Checkpoint : mode == "keep" ? mg::BuildMode::KeepWeights : mg::BuildMode::Fresh;
        if (mode != "keep") app.session->ClearLog();
        if (!app.session->StartBuild(DocFrom(app, body), m)) throw mg::GraphError("busy: wait for the current job or press stop");
        return {{"started", true}};
    }));
    svr.Post("/api/generate", Api([&](const auto &, const json &body) -> json
    {
        mg::GenerateOptions o;
        o.prompt = body.value("prompt", "");
        o.maxTokens = body.value("maxTokens", 64);
        o.temperature = body.value("temperature", 0.0f);
        o.chat = body.value("chat", false);
        o.system = body.value("system", "");
        o.seed = body.value("seed", uint64_t{0});
        if (!app.session->StartGenerate(o)) throw mg::GraphError("busy: wait for the current job or press stop");
        return {{"started", true}};
    }));
    svr.Post("/api/stop", Api([&](const auto &, const json &) -> json
    {
        app.session->RequestStop();
        return {{"stopping", true}};
    }));
    svr.Get("/api/status", Api([&](const auto &, const json &) -> json
    {
        const mg::Session::Snapshot s = app.session->GetSnapshot();
        return {{"busy", s.busy}, {"ready", s.ready}, {"phase", s.phase}, {"log", s.log}, {"output", s.output}, {"prompt", s.prompt},
                {"tokensPerSecond", s.tokensPerSecond}, {"parameters", s.parameters}, {"buildCount", s.buildCount},
                {"registryVersion", app.registry.Version()}, {"scriptGeneration", app.scriptGeneration.load()}};
    }));

    // ---- python scripts ----
    svr.Get("/api/scripts", Api([&](const auto &, const json &) -> json
    {
        std::lock_guard lock(app.hostMutex);
        json list = json::array();
        for (const auto &s : app.host->Statuses())
        {
            list.push_back({{"file", s.file.filename().string()}, {"type", s.type}, {"ok", s.ok}, {"error", s.error}, {"reloads", s.reloads}});
        }
        return {{"scripts", list}, {"dir", app.scripts.string()}};
    }));
    svr.Get(R"(/api/scripts/(.+))", Api([&](const httplib::Request &req, const json &) -> json
    {
        const fs::path p = SafePath(app.scripts, req.matches[1], ".py");
        return {{"file", p.filename().string()}, {"text", ReadText(p)}};
    }));
    svr.Put(R"(/api/scripts/(.+))", Api([&](const httplib::Request &req, const json &body) -> json
    {
        const fs::path p = SafePath(app.scripts, req.matches[1], ".py");
        WriteText(p, body.at("text").get<std::string>());
        return {{"saved", p.filename().string()}};
    }));
    svr.Post("/api/scripts", Api([&](const auto &, const json &body) -> json
    {
        const std::string name = body.at("name").get<std::string>();
        static const std::regex ok(R"(^[A-Za-z_][A-Za-z0-9_]*$)");
        if (!std::regex_match(name, ok)) throw mg::GraphError("node names must be identifiers, e.g. MyNode");
        const fs::path p = mg::script::ScriptHost::CreateStarter(app.scripts, name);
        return {{"file", p.filename().string()}};
    }));
    svr.Delete(R"(/api/scripts/(.+))", Api([&](const httplib::Request &req, const json &) -> json
    {
        fs::remove(SafePath(app.scripts, req.matches[1], ".py"));
        return {{"deleted", true}};
    }));

    // ---- the web app ----
    if (fs::is_directory(app.web))
    {
        svr.set_mount_point("/", app.web.string());
    }
    else
    {
        svr.Get("/", [&](const auto &, httplib::Response &res)
        {
            res.set_content("ModelGraph API is running, but the web UI is not built.\n"
                            "Run:  cd web && npm install && npm run build\n", "text/plain");
        });
    }

    const std::string url = "http://127.0.0.1:" + std::to_string(port);
    std::cout << "ModelGraph\n  project  " << app.root.string() << "\n  editor   " << url << "\n" << std::flush;
    if (browser) OpenBrowser(url);
    if (!svr.listen("127.0.0.1", port))
    {
        std::cerr << "error: cannot listen on port " << port << " (already in use?)\n";
    }
    app.running = false;
    watcher.join();
    return 0;
}
