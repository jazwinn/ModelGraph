// mg_cli: headless companion to the editor.
//
//   mg_cli check    <graph.json>                      shapes, parameter counts, errors
//   mg_cli run      <graph.json> [options] <prompt>   build and generate
//   mg_cli template <gpt2|llama3.2-1b|tiny> <out.json> write a starter graph
//
// run options: --tokens N  --temp T  --chat  --fresh  --device cpu|cuda

#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

#include "Graph/Analyze.h"
#include "Graph/Templates.h"
#include "Script/ScriptHost.h"
#include "Session/Session.h"

namespace
{
    int Usage()
    {
        std::cerr << "usage:\n"
                     "  mg_cli check    <graph.json>\n"
                     "  mg_cli run      <graph.json> [--tokens N] [--temp T] [--chat] [--fresh] [--device cpu|cuda] <prompt>\n"
                     "  mg_cli template <gpt2|llama3.2-1b|tiny> <out.json>\n";
        return 2;
    }

    void LoadScripts(mg::script::ScriptHost &host, const mg::Document &doc)
    {
        host.AddDirectory(doc.path.parent_path() / "scripts");
        host.AddDirectory(doc.path.parent_path().parent_path() / "scripts");
        host.Poll();
        for (const auto &s : host.Statuses())
        {
            if (!s.ok) std::cerr << "script " << s.file.filename().string() << ": " << s.error << "\n";
        }
    }

    void PrintGraph(const mg::Document &doc, const std::string &title, const mg::Graph &g, const mg::Analysis &a)
    {
        std::cout << "== " << title << "\n";
        for (const int id : a.order)
        {
            const mg::Node &n = *g.Find(id);
            std::cout << "  " << n.name << " (" << n.type << ")";
            if (auto it = a.outputs.find(id); it != a.outputs.end())
            {
                for (const auto &t : it->second) std::cout << " -> " << t.Text();
            }
            if (auto it = a.params.find(id); it != a.params.end() && it->second) std::cout << "  params " << it->second;
            if (auto it = a.errors.find(id); it != a.errors.end()) std::cout << "  ERROR: " << it->second;
            std::cout << "\n";
        }
        for (const auto &e : a.graphErrors) std::cout << "  ERROR: " << e << "\n";
        (void)doc;
    }
}

int main(int argc, char **argv)
{
    std::vector<std::string> args(argv + 1, argv + argc);
    if (args.empty()) return Usage();

    try
    {
        mg::Registry registry;
        mg::RegisterBuiltins(registry);
        mg::script::ScriptHost scripts(registry);

        if (args[0] == "template" && args.size() == 3)
        {
            mg::Document doc = mg::MakeTemplate(args[1]);
            doc.Save(args[2]);
            std::cout << "wrote " << args[2] << "\n";
            return 0;
        }

        if (args[0] == "check" && args.size() == 2)
        {
            const mg::Document doc = mg::Document::Load(args[1]);
            LoadScripts(scripts, doc);
            const mg::Analysis main = mg::Analyze(doc, doc.main, registry);
            PrintGraph(doc, "main", doc.main, main);
            for (const auto &[name, g] : doc.subgraphs)
            {
                PrintGraph(doc, "subgraph " + name, g, mg::Analyze(doc, g, registry));
            }
            std::cout << "total parameters: " << main.totalParams << "\n";
            return main.Ok() ? 0 : 1;
        }

        if (args[0] == "run" && args.size() >= 3)
        {
            mg::Document doc = mg::Document::Load(args[1]);
            LoadScripts(scripts, doc);
            mg::GenerateOptions options;
            mg::BuildMode mode = doc.model.checkpoint.empty() ? mg::BuildMode::Fresh : mg::BuildMode::Checkpoint;
            std::string prompt;
            for (size_t i = 2; i < args.size(); ++i)
            {
                const std::string &a = args[i];
                if (a == "--tokens" && i + 1 < args.size()) options.maxTokens = std::stoi(args[++i]);
                else if (a == "--temp" && i + 1 < args.size()) options.temperature = std::stof(args[++i]);
                else if (a == "--device" && i + 1 < args.size()) doc.model.device = args[++i];
                else if (a == "--chat") options.chat = true;
                else if (a == "--system" && i + 1 < args.size()) doc.model.systemPrompt = args[++i];
                else if (a == "--fresh") mode = mg::BuildMode::Fresh;
                else prompt += (prompt.empty() ? "" : " ") + a;
            }
            options.prompt = prompt;
            options.system = doc.model.systemPrompt;

            mg::Session session(registry);
            session.Build(doc, mode);
            std::cerr << session.GetSnapshot().log;
            session.ClearLog();
            std::cout << prompt << std::flush;
            session.Generate(options, [](const std::string &piece) { std::cout << piece << std::flush; });
            std::cout << "\n";
            std::cerr << session.GetSnapshot().log;
            return 0;
        }
        return Usage();
    }
    catch (const std::exception &e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
