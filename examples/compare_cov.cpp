// Archivo: compare_cov.cpp
#include "ibex_CovOptimData.h"
#include <iostream>

void compare_cov_files(const char* file1, const char* file2, const char* file3) {
    try {
        ibex::CovOptimData data1(file1);
        ibex::CovOptimData data2(file2);
        ibex::CovOptimData data3(file3);

        std::cout << "=== COMPARACIÓN DE ARCHIVOS COV ===" << std::endl;
        std::cout << "Métrica\t\t\tArchivo 1 (" << file1 << ")\tArchivo 2 (" << file2 << ")\tArchivo 3 (" << file3 << ")" << std::endl;
        std::cout << "---------------------------------------------------------------------------------------------------------" << std::endl;
        std::cout << "Tiempo (s):\t\t" << data1.time() << "\t\t\t" << data2.time() << "\t\t\t" << data3.time() << std::endl;
        std::cout << "Nº Celdas/Nodos:\t" << data1.nb_cells() << "\t\t\t" << data2.nb_cells() << "\t\t\t" << data3.nb_cells() << std::endl;
        std::cout << "Cota Inferior (uplo):\t" << data1.uplo() << "\t\t\t" << data2.uplo() << "\t\t\t" << data3.uplo() << std::endl;
        std::cout << "Cota Superior (loup):\t" << data1.loup() << "\t\t\t" << data2.loup() << "\t\t\t" << data3.loup() << std::endl;
        std::cout << "Cajas guardadas:\t" << data1.size() << "\t\t\t" << data2.size() << "\t\t\t" << data3.size() << std::endl;
    } catch (std::exception& e) {
        std::cerr << "Error al abrir o procesar los archivos: " << e.what() << std::endl;
    }
}

int main(int argc, char** argv) {
    // Se actualiza la validación para requerir 3 archivos (el nombre del programa es el argumento 0)
    if (argc < 4) {
        std::cout << "Uso: " << argv[0] << " <archivo1.cov> <archivo2.cov> <archivo3.cov>" << std::endl;
        return 1;
    }

    compare_cov_files(argv[1], argv[2], argv[3]);
    return 0;
}