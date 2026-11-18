#pragma once

#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "Graph/Types.h"
#include "Graph/Expr.h"

namespace mg
{
    struct Node
    {
        int id = 0;
        std::string type;
        std::string name;   // unique in its graph; also the parameter-name prefix
        AttrMap attrs;
        float x = 0, y = 0;
        nlohmann::json ui;   // editor-only state (e.g. whether a nest is expanded); round-tripped as is
    };

    struct Endpoint
    {
        int node = 0;
        std::string port;
        bool operator==(const Endpoint &o) const { return node == o.node && port == o.port; }
    };

    struct Link
    {
        int id = 0;
        Endpoint from;  // an output port
        Endpoint to;    // an input port
    };

    struct Graph
    {
        std::vector<Node> nodes;
        std::vector<Link> links;
        int nextId = 1;

        Node *Find(int id);
        const Node *Find(int id) const;
        const Node *FindByName(const std::string &name) const;
        Node &Add(std::string type, std::string name, AttrMap attrs = {}, float x = 0, float y = 0);
        // Connects from.port -> to.port, replacing whatever fed that input before.
        Link &Connect(int fromNode, const std::string &fromPort, int toNode, const std::string &toPort);
        void Remove(int nodeId);
        void RemoveLink(int linkId);
        const Link *LinkInto(int node, const std::string &port) const;
        std::string UniqueName(const std::string &stem) const;
    };

    struct RenameRuleSpec
    {
        std::string from;
        std::string to;
    };

    struct ModelSettings
    {
        std::string checkpoint;         // .safetensors, relative to the document
        std::string format = "raw";     // raw | gpt2 | llama
        std::vector<RenameRuleSpec> rules;
        std::string tokenizer;          // tokenizer.json, relative to the document
        std::string chatTemplate;       // tokenizer_config.json (optional)
        std::string systemPrompt = "You are a helpful assistant.";   // used in chat mode; empty = none
        std::vector<int64_t> eos;
        std::string device = "cuda";    // cpu | cuda
        int64_t contextSize = 2048;
    };

    struct Document
    {
        std::string name = "Untitled";
        std::map<std::string, AttrValue> variables;  // hyper-parameters referenced as $name
        Graph main;
        std::map<std::string, Graph> subgraphs;
        ModelSettings model;
        std::filesystem::path path;                 // where it was loaded from / saved to

        Variables NumericVariables() const;
        std::filesystem::path Resolve(const std::string &relative) const;

        nlohmann::json ToJson() const;
        static Document FromJson(const nlohmann::json &j);
        static Document Load(const std::filesystem::path &file);
        void Save(const std::filesystem::path &file);
    };

    nlohmann::json AttrToJson(const AttrValue &value);
    AttrValue AttrFromJson(const nlohmann::json &j);
}
