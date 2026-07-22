// Archivo: compare_cov.cpp
#include "ibex_CovOptimData.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <fstream>

bool file_exists(const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

void print_separator() {
    std::cout << std::string(145, '-') << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Uso: " << argv[0] << " <carpeta/*.cov>" << std::endl;
        std::cout << "Ejemplo: " << argv[0] << " ../2results/*.cov" << std::endl;
        return 1;
    }

    std::cout << "\n=== TABLA RECOPILATORIA COMPLETA (IBEX vs MODELO RL Y SOLO EVAL) ===" << std::endl;
    print_separator();
    std::cout << std::left 
              << std::setw(22) << "Problema" 
              << "| " << std::setw(20) << "Tiempo (IBEX | RL)" 
              << "| " << std::setw(20) << "Nodos (IBEX | RL)" 
              << "| " << std::setw(32) << "Cota Inf/uplo (IBEX | RL)" 
              << "| " << std::setw(32) << "Cota Sup/loup (IBEX | RL)" << std::endl;
    print_separator();

    int total_processed = 0;

    for (int i = 1; i < argc; ++i) {
        std::string current_arg = argv[i];
        std::string file_ibex = "";
        std::string file_eval = "";
        std::string prob_name = "";

        if (current_arg.find("_eval.cov") != std::string::npos) {
            file_eval = current_arg;
            file_ibex = current_arg;
            size_t pos = file_ibex.find("_eval");
            if (pos != std::string::npos) {
                file_ibex.erase(pos, 5);
            }
        } 
        else {
            file_ibex = current_arg;
            file_eval = current_arg;
            size_t pos = file_eval.find_last_of('.');
            if (pos != std::string::npos) {
                file_eval.insert(pos, "_eval");
            } else {
                file_eval += "_eval.cov";
            }
            
        }

        bool has_ibex = file_exists(file_ibex);
        bool has_eval = file_exists(file_eval);
        if (!file_exists(current_arg)) continue;
        if (current_arg.find("_eval.cov") == std::string::npos && has_eval) {
            continue; 
        }

        // Obtener nombre limpio del problema
        std::string target_file = has_eval ? file_eval : file_ibex;
        prob_name = target_file;
        size_t slash_pos = prob_name.find_last_of('/');
        if (slash_pos != std::string::npos) {
            prob_name = prob_name.substr(slash_pos + 1);
        }
        size_t dot_pos = prob_name.find_last_of('.');
        if (dot_pos != std::string::npos) {
            prob_name = prob_name.substr(0, dot_pos);
        }
        size_t eval_pos = prob_name.find("_eval");
        if (eval_pos != std::string::npos) {
            prob_name = prob_name.substr(0, eval_pos);
        }

        try {
            total_processed++;
            std::cout << std::left << std::setw(22) << prob_name << "| ";

            if (has_ibex && has_eval) {
                ibex::CovOptimData data_ibex(file_ibex.c_str());
                ibex::CovOptimData data_eval(file_eval.c_str());

                std::cout << std::setw(8) << data_ibex.time() << " | " << std::setw(9) << data_eval.time() << "| ";
                std::cout << std::setw(8) << data_ibex.nb_cells() << " | " << std::setw(9) << data_eval.nb_cells() << "| ";
                std::cout << std::setw(13) << data_ibex.uplo() << " | " << std::setw(14) << data_eval.uplo() << "| ";
                std::cout << std::setw(13) << data_ibex.loup() << " | " << std::setw(14) << data_eval.loup() << std::endl;

            } else if (has_eval && !has_ibex) {
                ibex::CovOptimData data_eval(file_eval.c_str());

                std::cout << std::setw(8) << "N/A" << " | " << std::setw(9) << data_eval.time() << "| ";
                std::cout << std::setw(8) << "N/A" << " | " << std::setw(9) << data_eval.nb_cells() << "| ";
                std::cout << std::setw(13) << "N/A" << " | " << std::setw(14) << data_eval.uplo() << "| ";
                std::cout << std::setw(13) << "N/A" << " | " << std::setw(14) << data_eval.loup() << " (Solo RL)" << std::endl;

            } else if (has_ibex && !has_eval) {
                ibex::CovOptimData data_ibex(file_ibex.c_str());

                std::cout << std::setw(8) << data_ibex.time() << " | " << std::setw(9) << "N/A" << "| ";
                std::cout << std::setw(8) << data_ibex.nb_cells() << " | " << std::setw(9) << "N/A" << "| ";
                std::cout << std::setw(13) << data_ibex.uplo() << " | " << std::setw(14) << "N/A" << "| ";
                std::cout << std::setw(13) << data_ibex.loup() << " | " << std::setw(14) << "N/A" << " (Solo IBEX)" << std::endl;
            }

        } catch (...) {
        }
    }

    print_separator();
    std::cout << "Total de filas impresas en la tabla: " << total_processed << std::endl;
    return 0;
}