//                                  I B E X
// File        : ibex_Optimizer.cpp (JSON + ACTIVACIÓN + ESTADÍSTICAS BISECTORES)
//============================================================================

#include "ibex_Optimizer.h"
#include "ibex_Timer.h"
#include "ibex_Function.h"
#include "ibex_NoBisectableVariableException.h"
#include "ibex_BxpOptimData.h"
#include "ibex_CovOptimData.h"
#include "ibex_CellBeamSearch.h"
#include "ibex_LoupFinderDefault.h"
#include "ibex_SmearFunction.h"
#include "ibex_ExtendedSystem.h"
#include "ibex_OptimLargestFirst.h"
#include "ibex_System.h"
#include <float.h>
#include <stdlib.h>
#include <vector>
#include <queue>
#include <iomanip>
#include <chrono>
#include <limits> 
#include <cmath>
#include <sstream> 
#include <map> // [NUEVO] Para mapear nombres

using namespace std;

// ============================================================================
//  CONFIGURACIÓN Y VARIABLES GLOBALES
// ============================================================================
const int MIN_DEPTH = 5;          
const int STAGNATION_LIMIT = 10;  

// Métricas de Tiempo
static double g_python_time = 0.0;
static long g_wall_time = 0;
static std::chrono::time_point<std::chrono::high_resolution_clock> g_start_wall_time;

// [NUEVO] Contadores de Bisectores (0..5 + Default)
// 0:LSMEAR, 1:LF, 2:RR, 3:SM, 4:SS, 5:SSR, 6:Default(Sin Red)
static long g_bisector_counts[7] = {0, 0, 0, 0, 0, 0, 0}; 

// ============================================================================
//         COMUNICACIÓN CON MODELO
// ============================================================================
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string>
#include <iostream>

static int model_socket = -1;
static bool socket_connected = false;

void close_model_connection() {
    if (socket_connected && model_socket != -1) {
        close(model_socket); socket_connected = false; model_socket = -1;
    }
}

bool connect_to_model_server(const std::string& host = "127.0.0.1", int port = 8888) {
    if (socket_connected) return true;
    model_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (model_socket == -1) return false;
    
    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(port);
    if (inet_pton(AF_INET, host.c_str(), &serv_addr.sin_addr) <= 0) { close(model_socket); return false; }
    if (connect(model_socket, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) { close(model_socket); return false; }
    socket_connected = true;
    return true;
}

std::string call_python_model(const std::string& message) {
    if (!socket_connected && !connect_to_model_server()) return "";
    if (send(model_socket, message.c_str(), message.length(), 0) == -1) { close_model_connection(); return ""; }
    char buffer[4096] = {0};
    if (recv(model_socket, buffer, sizeof(buffer) - 1, 0) > 0) return std::string(buffer);
    else { close_model_connection(); return ""; }
}

int parse_decision_from_json(const std::string& json_str) {
    std::string key = "\"decision\":";
    size_t pos = json_str.find(key);
    if (pos == std::string::npos) {
        key = "\"decision\": ";
        pos = json_str.find(key);
        if (pos == std::string::npos) return -1;
    }
    size_t start = pos + key.length();
    try { return std::stoi(json_str.substr(start)); } catch (...) { return -1; }
}

// ============================================================================
//  PREPROCESAMIENTO
// ============================================================================
double const UMBRAL_CERO = 1e-7;
enum class FeatureType { LbFObj, UbFObj, BiggerDiam, GenericFeature };

double preprocess_feature(double val, FeatureType type) {
    if (std::isnan(val)) return 0.0;
    if (std::abs(val) < UMBRAL_CERO) val = 0.0;
    switch (type) {
        case FeatureType::LbFObj: return std::max(-1e7, std::min(val, 1e7));
        case FeatureType::UbFObj: return std::min(val, 1e6);
        case FeatureType::BiggerDiam: return std::min(val, 1e6);
        default: if (std::isinf(val)) return (val > 0) ? DBL_MAX : -DBL_MAX; break;
    }
    return val;
}

FeatureType string_to_feature_type(const std::string& feature_name) {
    if (feature_name == "lb_f_obj") return FeatureType::LbFObj;
    if (feature_name == "ub_f_obj") return FeatureType::UbFObj;
    if (feature_name == "bigger_diam") return FeatureType::BiggerDiam;
    return FeatureType::GenericFeature;
}

double preprocess_feature(double raw_value, const std::string& feature_name) {
    return preprocess_feature(raw_value, string_to_feature_type(feature_name));
}

namespace ibex {

void Optimizer::write_ext_box(const IntervalVector& box, IntervalVector& ext_box) {
    int i2=0; for (int i=0; i<n; i++,i2++) { if (i2==goal_var) i2++; ext_box[i2]=box[i]; }
}

void Optimizer::read_ext_box(const IntervalVector& ext_box, IntervalVector& box) {
    int i2=0; for (int i=0; i<n; i++,i2++) { if (i2==goal_var) i2++; box[i]=ext_box[i2]; }
}

Optimizer::Optimizer(int n, Ctc& ctc, Bsc& bsc, LoupFinder& finder, CellBufferOptim& buffer,
        int goal_var, double eps_x, double rel_eps_f, double abs_eps_f, bool enable_statistics) :
        n(n), goal_var(goal_var), ctc(ctc), bsc(bsc), loup_finder(finder), buffer(buffer),
        eps_x(n, eps_x), rel_eps_f(rel_eps_f), abs_eps_f(abs_eps_f), trace(0), timeout(-1), 
        extended_COV(true), anticipated_upper_bounding(true), status(SUCCESS),
        uplo(NEG_INFINITY), uplo_of_epsboxes(POS_INFINITY), loup(POS_INFINITY),
        loup_point(IntervalVector::empty(n)), initial_loup(POS_INFINITY), loup_changed(false),
        time(0), nb_cells(0), cov(NULL) {
    if (enable_statistics) {
        statistics = new Statistics();
        bsc.enable_statistics(*statistics, "Bsc"); ctc.enable_statistics(*statistics, "Ctc"); loup_finder.enable_statistics(*statistics, "LoupFinder"); 
    } else statistics = NULL;
}

Optimizer::Optimizer(OptimizerConfig& config) : Optimizer(config.nb_var(), config.get_ctc(), config.get_bsc(), config.get_loup_finder(), config.get_cell_buffer(), config.goal_var(), OptimizerConfig::default_eps_x, config.get_rel_eps_f(), config.get_abs_eps_f(), config.with_statistics()) {
    (Vector&) eps_x = config.get_eps_x(); trace = config.get_trace(); timeout = config.get_timeout(); extended_COV = config.with_extended_cov(); anticipated_upper_bounding = config.with_anticipated_upper_bounding();
}

Optimizer::~Optimizer() {
    if (cov) delete cov; if (statistics) delete statistics; close_model_connection(); 
}

double Optimizer::compute_ymax() {
    if (anticipated_upper_bounding) {
        double ymax = loup>0 ? 1/(1+rel_eps_f)*loup : 1/(1-rel_eps_f)*loup;
        if (loup - abs_eps_f < ymax) ymax = loup - abs_eps_f;
        return next_float(ymax);
    } else return loup;
}

bool Optimizer::update_loup(const IntervalVector& box, BoxProperties& prop) {
    try {
        pair<IntervalVector,double> p=loup_finder.find(box,loup_point,loup,prop);
        loup_point = p.first; loup = p.second;
        if (trace) std::cout << "\033[32m loup= " << loup << "\033[0m" << endl;
        return true;
    } catch(LoupFinder::NotFound&) { return false; }
}

void Optimizer::update_uplo() {
    double new_uplo=POS_INFINITY;
    if (! buffer.empty()) {
        new_uplo= buffer.minimum();
        if (new_uplo < uplo_of_epsboxes) {
            if (new_uplo > uplo) { uplo = new_uplo; if (trace) std::cout << "\033[33m uplo= " << uplo << "\033[0m" << endl; }
        } else uplo = uplo_of_epsboxes;
    } else if (buffer.empty() && loup != POS_INFINITY) {
        new_uplo=compute_ymax(); double m = (new_uplo < uplo_of_epsboxes) ? new_uplo :  uplo_of_epsboxes; if (uplo < m) uplo = m; 
    }
}

void Optimizer::update_uplo_of_epsboxes(double ymin) { if (uplo_of_epsboxes > ymin) uplo_of_epsboxes = ymin; }

void Optimizer::handle_cell(Cell& c) { contract_and_bound(c); if (c.box.is_empty()) delete &c; else buffer.push(&c); }

void Optimizer::contract_and_bound(Cell& c) {
    Interval& y=c.box[goal_var]; double ymax;
    if (loup==POS_INFINITY) ymax = POS_INFINITY; else ymax = compute_ymax()+1.e-15;
    y &= Interval(NEG_INFINITY,ymax);
    if (y.is_empty()) { c.box.set_empty(); return; } 
    else { c.prop.update(BoxEvent(c.box,BoxEvent::CONTRACT,BitSet::singleton(n+1,goal_var))); }

    ContractContext context(c.prop);
    if (c.bisected_var!=-1) { context.impact.clear(); context.impact.add(c.bisected_var); context.impact.add(goal_var); }
    ctc.contract(c.box, context);
    if (c.box.is_empty()) return;

    IntervalVector tmp_box(n); read_ext_box(c.box,tmp_box);
    c.prop.update(BoxEvent(c.box,BoxEvent::CHANGE));
    bool loup_ch=update_loup(tmp_box, c.prop);
    if (loup_ch) { y &= Interval(NEG_INFINITY,compute_ymax()); c.prop.update(BoxEvent(c.box,BoxEvent::CONTRACT,BitSet::singleton(n+1,goal_var))); }
    loup_changed |= loup_ch;
    if (y.is_empty()) { c.box.set_empty(); return; }
    if (((tmp_box.diam()-eps_x).max()<=0 && y.diam() <=abs_eps_f) || !c.box.is_bisectable()) { update_uplo_of_epsboxes(y.lb()); c.box.set_empty(); return; }
    if (tmp_box.is_empty()) { c.box.set_empty(); } else { write_ext_box(tmp_box,c.box); }
}

Optimizer::Status Optimizer::optimize(const IntervalVector& init_box, double obj_init_bound) { start(init_box, obj_init_bound); return optimize(); }
Optimizer::Status Optimizer::optimize(const CovOptimData& data, double obj_init_bound) { start(data, obj_init_bound); return optimize(); }
Optimizer::Status Optimizer::optimize(const char* cov_file, double obj_init_bound) { CovOptimData data(cov_file); start(data, obj_init_bound); return optimize(); }

void Optimizer::start(const IntervalVector& init_box, double obj_init_bound) {
    loup=obj_init_bound; buffer.contract(loup); uplo=NEG_INFINITY; uplo_of_epsboxes=POS_INFINITY; nb_cells=0; buffer.flush();
    Cell* root=new Cell(IntervalVector(n+1)); write_ext_box(init_box, root->box);
    bsc.add_property(init_box, root->prop); ctc.add_property(init_box, root->prop); buffer.add_property(init_box, root->prop); loup_finder.add_property(init_box, root->prop);
    loup_changed=false; initial_loup=obj_init_bound; loup_point = init_box; time=0;
    
    // Inicializar Globales
    g_python_time = 0.0; g_wall_time = 0; g_start_wall_time = std::chrono::high_resolution_clock::now();
    for(int k=0; k<7; k++) g_bisector_counts[k] = 0; // Reset contadores
    
    connect_to_model_server();
    if (cov) delete cov; cov = new CovOptimData(extended_COV? n+1 : n, extended_COV);
    cov->data->_optim_time = 0; cov->data->_optim_nb_cells = 0;
    handle_cell(*root);
}

void Optimizer::start(const CovOptimData& data, double obj_init_bound) {
    loup=obj_init_bound; buffer.contract(loup); uplo=data.uplo(); loup=data.loup(); loup_point=data.loup_point(); uplo_of_epsboxes=POS_INFINITY; nb_cells=0; buffer.flush();
    for (size_t i=loup_point.is_empty()? 0 : 1; i<data.size(); i++) {
        IntervalVector box(n+1);
        if (data.is_extended_space()) box = data[i]; else { write_ext_box(data[i], box); box[goal_var] = Interval(uplo,loup); ctc.contract(box); if (box.is_empty()) continue; }
        Cell* cell=new Cell(box); buffer.add_property(box, cell->prop); bsc.add_property(box, cell->prop); ctc.add_property(box, cell->prop); loup_finder.add_property(box, cell->prop); buffer.push(cell);
    }
    loup_changed=false; initial_loup=obj_init_bound; time=0;
    g_python_time = 0.0; g_wall_time = 0; g_start_wall_time = std::chrono::high_resolution_clock::now();
    for(int k=0; k<7; k++) g_bisector_counts[k] = 0;
    connect_to_model_server();
    if (cov) delete cov; cov = new CovOptimData(extended_COV? n+1 : n, extended_COV);
    cov->data->_optim_time = data.time(); cov->data->_optim_nb_cells = data.nb_cells();
}

Optimizer::Status Optimizer::optimize() {
    Timer timer; auto now = std::chrono::high_resolution_clock::now(); timer.start(); Timer python_timer;
    pid_t pid1 = getpid(); update_uplo();

    try {
        CellBeamSearch * thebuffer = dynamic_cast<CellBeamSearch*>(&buffer);
        LoupFinderDefault * lfd = dynamic_cast<LoupFinderDefault*>(&loup_finder);
        queue<Cell*> aux; aux.push(thebuffer->top()); 
        
        double prec = 1e-7;
        OptimLargestFirst bisector_olf(goal_var, true, prec, 0.5);
        RoundRobin bisector_rr(prec, 0.5);
        System system = lfd->finder_x_taylor.sys;
        SmearMax bisector_sm(system, prec);
        SmearSum bisector_ss(system, prec);
        SmearSumRelative bisector_ssr(system, prec);

        int iterations_without_improvement = 0;
        double last_uplo = uplo;

        while (!thebuffer->empty()) {
            loup_changed=false;
            Cell *c = thebuffer->top();
            
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - g_start_wall_time).count();
            if (elapsed >= 1800) { status = TIME_OUT; g_wall_time = elapsed; goto end; }

            // =========================================================
            //  ESTRATEGIA DE ACTIVACIÓN
            // =========================================================
            bool call_neural_network = false;

            if (uplo > last_uplo) {
                iterations_without_improvement = 0;
                last_uplo = uplo;
            } else {
                iterations_without_improvement++;
            }

            if (c->depth >= MIN_DEPTH) {
                if (iterations_without_improvement >= STAGNATION_LIMIT) {
                    call_neural_network = true;
                }
            }
            
            Bsc* chosen_bsc = &bsc; 
            int selected_heuristic_id = 6; // 6 = Default (No Red)

            if (call_neural_network) {
                // --- CONSTRUIR JSON ---
                int variables = n;
                int restricciones = lfd->finder_x_taylor.sys.nb_ctr;
                int depth = c->depth;
                double ub_f_obj = c->box[goal_var].ub();
                double lb_f_obj = c->box[goal_var].lb();
                double bigger_diam = c->box.max_diam();
                double diametro_pequeno = c->box.diam().min();

                lb_f_obj = preprocess_feature(lb_f_obj, "lb_f_obj");
                ub_f_obj = preprocess_feature(ub_f_obj, "ub_f_obj");
                bigger_diam = preprocess_feature(bigger_diam, "bigger_diam");
                diametro_pequeno = preprocess_feature(diametro_pequeno, "bigger_diam");

                std::stringstream ss;
                ss << std::fixed << std::setprecision(6);
                ss << "{\"features\": [" 
                   << variables << "," << restricciones << "," << depth << ","
                   << ub_f_obj << "," << lb_f_obj << "," << bigger_diam << "," 
                   << diametro_pequeno 
                   << "]}"; 

                python_timer.start();
                std::string response = call_python_model(ss.str());
                python_timer.stop();
                g_python_time += python_timer.get_time();

                int model_decision = parse_decision_from_json(response);
                if (model_decision == -1) model_decision = 0; // Fallback

                switch (model_decision) {
                    case 0: chosen_bsc = &bsc; break;          
                    case 1: chosen_bsc = &bisector_olf; break; 
                    case 2: chosen_bsc = &bisector_rr; break;  
                    case 3: chosen_bsc = &bisector_sm; break;  
                    case 4: chosen_bsc = &bisector_ss; break;  
                    case 5: chosen_bsc = &bisector_ssr; break; 
                }
                
                selected_heuristic_id = model_decision; // Guardar ID para estadística
                // std::cout << "Red Activada. Bisector: " << model_decision << std::endl;
                std::cout << "Red Activada (" << iterations_without_improvement << " iters sin mejora). Bisector: "<< model_decision << std::endl;
            } 
            
            // [NUEVO] Registrar Estadística
            if (selected_heuristic_id >= 0 && selected_heuristic_id <= 6) {
                g_bisector_counts[selected_heuristic_id]++;
            }

            // =========================================================
            //  FIN SELECCIÓN
            // =========================================================
            
            thebuffer->pop(); 
            if (trace>=2) { cout << " pop " << *c << endl; cout << " depth " << c->depth << endl; }
            nb_cells++;
            pair<Cell*,Cell*> new_cells=chosen_bsc->bisect(*c); 
            if (trace>=2) { cout << " left " << *new_cells.first << endl; cout << " right " << *new_cells.second << endl; }
            delete c;
            if (new_cells.first) handle_cell(*new_cells.first);
            if (new_cells.second) handle_cell(*new_cells.second);
            update_uplo();
            if (loup_changed) { if (trace) cout << "\033[32m loup= " << loup << "\033[0m" << endl; buffer.contract(loup); }
            if (uplo>=loup) { status=SUCCESS; goto end; }
            if (trace) { cout << " nb_cells=" << nb_cells << " buffer size=" << buffer.size() << " uplo=" << uplo << " loup=" << loup << endl; }
            now = std::chrono::high_resolution_clock::now();
            if (timeout>0 && timer.get_time()>timeout) { status=TIME_OUT; goto end; }
        }
        status=SUCCESS;
    } catch (NoBisectableVariableException& ) { status=SUCCESS; }

end:
    timer.stop(); time=timer.get_time();
    auto end_time = std::chrono::high_resolution_clock::now();
    g_wall_time = std::chrono::duration_cast<std::chrono::seconds>(end_time - g_start_wall_time).count();
    if (cov) { cov->data->_optim_time += time; cov->data->_optim_nb_cells += nb_cells; }
    return status;
}

// [NUEVO] Función Reporte Mejorada con Tabla de Bisectores
void Optimizer::report() {
    cout << " optimization status:\t\t";
    if (status==SUCCESS) cout << "Success" << endl; else cout << "Time Out" << endl;
    cout << " number of cells:\t\t" << nb_cells << endl;
    cout << " cpu time used:\t\t\t" << time << "s" << endl;
    cout << " tiempo en llamadas a Python:\t" << g_python_time << "s (" 
         << fixed << setprecision(2) << (time > 0 ? (g_python_time/time)*100.0 : 0.0) << "%)" << endl;
    cout << " wall clock time:\t\t" << g_wall_time << "s" << endl;
    cout << " best bound (f*):\t\t" << loup << endl;
    
    if (!loup_point.is_empty()) cout << " best point (x*):\t\t" << loup_point << endl;
    else cout << " best point (x*):\t\t(Ninguna solución factible encontrada)" << endl;

    // --- TABLA DE ESTADÍSTICAS DE BISECTORES ---
    cout << endl << " === Heuristic Usage Statistics ===" << endl;
    cout << " ----------------------------------" << endl;
    cout << "  0 (LSMEAR): " << g_bisector_counts[0] << endl;
    cout << "  1 (LF)    : " << g_bisector_counts[1] << endl;
    cout << "  2 (RR)    : " << g_bisector_counts[2] << endl;
    cout << "  3 (SM)    : " << g_bisector_counts[3] << endl;
    cout << "  4 (SS)    : " << g_bisector_counts[4] << endl;
    cout << "  5 (SSR)   : " << g_bisector_counts[5] << endl;
    cout << "  6 (Default): " << g_bisector_counts[6] << " (Red inactiva)" << endl;
    cout << " ----------------------------------" << endl;
    // -------------------------------------------

    if (statistics) cout << endl << "  ===== Statistics ====" << endl << endl << *statistics << endl;
}

} // end namespace ibex