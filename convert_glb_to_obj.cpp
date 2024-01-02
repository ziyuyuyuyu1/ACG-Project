#include <iostream>
#include <experimental/filesystem>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/Exporter.hpp>
#include <assimp/postprocess.h>

namespace fs = std::experimental::filesystem;

void convertGlbToObj(const std::string& glbPath, const std::string& outputFolder) {
    std::string objPath = outputFolder + "/" + fs::path(glbPath).stem().string() + ".obj";
    std::cout << objPath << std::endl;
    if (fs::exists(objPath)) {
        std::cout << "Skipping conversion for '" << glbPath << "' - '" << objPath << "' already exists." << std::endl;
        return;
    }
    Assimp::Importer importer;
    try {
        if (!fs::exists(glbPath)) {
            std::cout << "Error: GLB file '" << glbPath << "' does not exist." << std::endl;
            return;
        }
        const aiScene *scene = importer.ReadFile(glbPath.c_str(),
            aiProcess_Triangulate | aiProcess_JoinIdenticalVertices | aiProcess_GenSmoothNormals | aiProcess_FlipUVs);
        std::cout << "here" << std::endl;
        if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
            std::cerr << "Error loading GLB file '" << glbPath << "': " << importer.GetErrorString() << std::endl;
            return;
        }

        // // Generate output file path with .obj extension
        // std::string objPath = outputFolder + "/" + fs::path(glbPath).stem().string() + ".obj";

        // Export as OBJ
        Assimp::Exporter exporter;
        exporter.Export(scene, "obj", objPath.c_str());

        std::cout << "Conversion successful: " << glbPath << " -> " << objPath << std::endl;
    } catch (const std::exception& ex) {
        // Catch and print any exceptions that occur during processing
        std::cerr << "Exception while processing GLB file '" << glbPath << "': " << ex.what() << std::endl;
    }
}

int main() {
    std::string glbFolderPath = "/share1/jialuo/objaverse";
    std::string outputFolderPath = "/share1/jialuo/objaverse_output";

    // Create the output folder if it doesn't exist
    fs::create_directories(outputFolderPath);

    // Iterate through all .glb files in the specified directory and its subdirectories
    for (const auto& entry : fs::recursive_directory_iterator(glbFolderPath)) {
        if (entry.path().extension() == ".glb") {
            convertGlbToObj(entry.path().string(), outputFolderPath);
        }
    }

    return 0;
}
