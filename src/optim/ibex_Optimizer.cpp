//                                  I B E X
// File        : ibex_Optimizer.cpp (RL DYNAMIC HYPER-HEURISTIC FOR NODE SELECTION)
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
#include <sys/socket.h> //Comunicación con el modelo Python
#include <arpa/inet.h>
#include <unistd.h>
#include <string>
#include <iostream>

using namespace std;

const double TERMINAL_BONUS_SUCCESS         =  5.0;  
const double TERMINAL_BONUS_TIMEOUT         = -5.0;  
const double TERMINAL_BONUS_UNREACHED_PREC  = -3.0;  
const double TERMINAL_BONUS_NEUTRAL         =  0.0;  

// Métricas de Tiempo para comunciación Osorio
static double g_python_time = 0.0;
static long g_wall_time = 0;
static std::chrono::time_point<std::chrono::high_resolution_clock> g_start_wall_time;

//acciones posibles: 0=Best-First / minLB, 1=LBvUB, 2=FeasibleDiving, 3=FeasibleDivingUB
static long g_action_counts[4] = {0, 0, 0, 0};

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

//Comunicación con el modelo Python
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

namespace ibex {
    
//int nodes_pruned_in_k_step = 0;  

// Variables para el estado del RL y el Diving
Cell* dive_next = nullptr; 
double old_gap = POS_INFINITY;
int current_macro_action = 0; 
int iterations_in_macro_step = 0;
const int K_ITERATIONS = 100;
/*
 * TODO: redundant with ExtendedSystem.
 */
void Optimizer::write_ext_box(const IntervalVector& box, IntervalVector& ext_box) {
    int i2=0;
    for (int i=0; i<n; i++,i2++) {
        if (i2==goal_var) i2++; // skip goal variable
        ext_box[i2]=box[i];
    }
}

void Optimizer::read_ext_box(const IntervalVector& ext_box, IntervalVector& box) {
    int i2=0;
    for (int i=0; i<n; i++,i2++) {
        if (i2==goal_var) i2++; // skip goal variable
        box[i]=ext_box[i2];
    }
}
Optimizer::Optimizer(int n, Ctc& ctc, Bsc& bsc, LoupFinder& finder,
        CellBufferOptim& buffer,
        int goal_var, double eps_x, double rel_eps_f, double abs_eps_f,
        bool enable_statistics) :
                                        n(n), goal_var(goal_var),
                                        ctc(ctc), bsc(bsc), loup_finder(finder), buffer(buffer),
                                        eps_x(n, eps_x), rel_eps_f(rel_eps_f), abs_eps_f(abs_eps_f),
                                        trace(0), timeout(-1), extended_COV(true), anticipated_upper_bounding(true),
                                        status(SUCCESS),
                                        uplo(NEG_INFINITY), uplo_of_epsboxes(POS_INFINITY), loup(POS_INFINITY),
                                        loup_point(IntervalVector::empty(n)), initial_loup(POS_INFINITY), loup_changed(false),
                                        time(0), nb_cells(0), cov(NULL) {

    if (trace) cout.precision(12);
    
    if (enable_statistics) {
        statistics = new Statistics();
        bsc.enable_statistics(*statistics, "Bsc"); 
        ctc.enable_statistics(*statistics, "Ctc"); 
        loup_finder.enable_statistics(*statistics, "LoupFinder"); 
    } else
        statistics = NULL;
}

Optimizer::Optimizer(OptimizerConfig& config) :
    Optimizer(
        config.nb_var(), 
        config.get_ctc(), 
        config.get_bsc(), 
        config.get_loup_finder(),
        config.get_cell_buffer(),
        config.goal_var(),
        OptimizerConfig::default_eps_x, 
        config.get_rel_eps_f(),
        config.get_abs_eps_f(),
        config.with_statistics()) {

    (Vector&) eps_x             = config.get_eps_x();
    trace                       = config.get_trace();
    timeout                     = config.get_timeout();
    extended_COV                = config.with_extended_cov();
    anticipated_upper_bounding  = config.with_anticipated_upper_bounding();
}

Optimizer::~Optimizer() {
    if (cov) delete cov;
    if (statistics) delete statistics;
}

double Optimizer::compute_ymax() {
    if (anticipated_upper_bounding) {
        double ymax = loup>0 ?
                1/(1+rel_eps_f)*loup
        :
                1/(1-rel_eps_f)*loup;

        if (loup - abs_eps_f < ymax)
            ymax = loup - abs_eps_f;
        return next_float(ymax);
    } else
        return loup;
}

bool Optimizer::update_loup(const IntervalVector& box, BoxProperties& prop) {

    try {
        pair<IntervalVector,double> p=loup_finder.find(box,loup_point,loup,prop);
        loup_point = p.first;
        loup = p.second;

        if (trace) {
            cout << "                    ";
            cout << "\033[32m loup= " << loup << "\033[0m" << endl;
        }
        return true;

    } catch(LoupFinder::NotFound&) {
        return false;
    }
}

void Optimizer::update_uplo() {
    double new_uplo=POS_INFINITY;

    if (! buffer.empty()) {
        new_uplo= buffer.minimum();
        if (new_uplo > loup && uplo_of_epsboxes > loup) {
            cout << " loup = " << loup << " new_uplo=" << new_uplo <<  " uplo_of_epsboxes=" << uplo_of_epsboxes << endl;
            ibex_error("optimizer: new_uplo>loup (please report bug)");
        }
        if (new_uplo < uplo) {
            cout << "uplo= " << uplo << " new_uplo=" << new_uplo << endl;
            ibex_error("optimizer: new_uplo<uplo (please report bug)");
        }

        if (new_uplo < uplo_of_epsboxes) {
            if (new_uplo > uplo) {
                uplo = new_uplo;
                if (trace)
                    cout << "\033[33m uplo= " << uplo << "\033[0m" << endl;
            }
        }
        else uplo = uplo_of_epsboxes;
    }
    else if (buffer.empty() && loup != POS_INFINITY) {
        new_uplo=compute_ymax(); 
        double m = (new_uplo < uplo_of_epsboxes) ? new_uplo :  uplo_of_epsboxes;
        if (uplo < m) uplo = m; 
    }
}

void Optimizer::update_uplo_of_epsboxes(double ymin) {
    assert (uplo_of_epsboxes >= uplo);
    assert(ymin >= uplo);
    if (uplo_of_epsboxes > ymin) {
        uplo_of_epsboxes = ymin;
        if (trace) {
            cout << " unprocessable tiny box: now uplo<=" << setprecision(12) <<  uplo_of_epsboxes << " uplo=" << uplo << endl;
        }
    }
}

void Optimizer::handle_cell(Cell& c) {
    contract_and_bound(c);
    if (c.box.is_empty()) {
        delete &c;
    } else {
        buffer.push(&c);
    }
}

void Optimizer::contract_and_bound(Cell& c) {
    Interval& y=c.box[goal_var];

    double ymax;
    if (loup==POS_INFINITY) ymax = POS_INFINITY;
    else ymax = compute_ymax()+1.e-15;

    y &= Interval(NEG_INFINITY,ymax);

    if (y.is_empty()) {
        c.box.set_empty();
        return;
    } else {
        c.prop.update(BoxEvent(c.box,BoxEvent::CONTRACT,BitSet::singleton(n+1,goal_var)));
    }

    ContractContext context(c.prop);
    if (c.bisected_var!=-1) {
        context.impact.clear();
        context.impact.add(c.bisected_var);
        context.impact.add(goal_var);
    }

    ctc.contract(c.box, context);
    if (c.box.is_empty()) return;

    IntervalVector tmp_box(n);
    read_ext_box(c.box,tmp_box);
    c.prop.update(BoxEvent(c.box,BoxEvent::CHANGE));

    bool loup_ch=update_loup(tmp_box, c.prop);

    if (loup_ch) {
        y &= Interval(NEG_INFINITY,compute_ymax());
        c.prop.update(BoxEvent(c.box,BoxEvent::CONTRACT,BitSet::singleton(n+1,goal_var)));
    }

    loup_changed |= loup_ch;

    if (y.is_empty()) { 
        c.box.set_empty();
        return;
    }

    if (((tmp_box.diam()-eps_x).max()<=0 && y.diam() <=abs_eps_f) || !c.box.is_bisectable()) {
        update_uplo_of_epsboxes(y.lb());
        c.box.set_empty();
        return;
    }

    if (tmp_box.is_empty()) {
        c.box.set_empty();
    } else {
        write_ext_box(tmp_box,c.box);
    }
}

Optimizer::Status Optimizer::optimize(const IntervalVector& init_box, double obj_init_bound) {
    start(init_box, obj_init_bound);
    return optimize();
}

Optimizer::Status Optimizer::optimize(const CovOptimData& data, double obj_init_bound) {
    start(data, obj_init_bound);
    return optimize();
}

Optimizer::Status Optimizer::optimize(const char* cov_file, double obj_init_bound) {
    CovOptimData data(cov_file);
    start(data, obj_init_bound);
    return optimize();
}

void Optimizer::start(const IntervalVector& init_box, double obj_init_bound) {
    loup=obj_init_bound;
    buffer.contract(loup);
    uplo=NEG_INFINITY;
    uplo_of_epsboxes=POS_INFINITY;
    nb_cells=0;
    buffer.flush();

    Cell* root=new Cell(IntervalVector(n+1));
    write_ext_box(init_box, root->box);

    bsc.add_property(init_box, root->prop);
    ctc.add_property(init_box, root->prop);
    buffer.add_property(init_box, root->prop);
    loup_finder.add_property(init_box, root->prop);

    loup_changed=false;
    initial_loup=obj_init_bound;

    loup_point = init_box; 
    time=0;

    if (cov) delete cov;
    cov = new CovOptimData(extended_COV? n+1 : n, extended_COV);
    cov->data->_optim_time = 0;
    cov->data->_optim_nb_cells = 0;

    handle_cell(*root);

    // =========================================================================
    // NUEVO: NOTIFICACIÓN DE ESTADO INICIAL PARA EL AGENTE RL
    // =========================================================================
    // 1. Limpiar variables del episodio anterior para no heredar basura
    iterations_in_macro_step = 0;
    dive_next = nullptr;
    old_gap = POS_INFINITY;
    current_macro_action = 0; 
    
    // 2. Calcular estado inicial exacto
    double current_gap = loup - uplo;
    
    double f1_log_nodos = std::log10(buffer.size() + 1.0);
    f1_log_nodos = f1_log_nodos / (f1_log_nodos + 2.0); 
    
    double f2_gap_pct;
    if (uplo == NEG_INFINITY || loup == POS_INFINITY) {
        f2_gap_pct = 1.0; 
    } else {
        double safe_gap = std::max(0.0, current_gap);
        double log_gap = std::log10(1.0 + safe_gap);
        f2_gap_pct = log_gap / (1.0 + log_gap); 
    }
    
    double f3_time_pct = 0.0; // Tiempo en 0
    double f4_depth = 0.0;    // Profundidad en 0 (Raíz)
    int f5_accion_ant = 0;    // Acción por defecto
    
    // 3. Empaquetar y enviar a Python
    std::stringstream ss;
    ss << "{ \"features\": [" 
       << f1_log_nodos << ", " << f2_gap_pct << ", " 
       << f3_time_pct << ", " << f4_depth << ", " << f5_accion_ant << "], "
       << "\"reward\": 0.0, "
       << "\"done\": false }\n";
       
    string json_response = call_python_model(ss.str());
    
    // 4. Configurar la primera acción elegida por el modelo para este episodio
    if (!json_response.empty()) {
        int decision = parse_decision_from_json(json_response);
        if (decision >= 0 && decision <= 3) {
            current_macro_action = decision;
        }
    }
    old_gap = current_gap;

}

void Optimizer::start(const CovOptimData& data, double obj_init_bound) {
    loup=obj_init_bound;
    buffer.contract(loup);
    uplo=data.uplo();
    loup=data.loup();
    loup_point=data.loup_point();
    uplo_of_epsboxes=POS_INFINITY;
    nb_cells=0;
    buffer.flush();

    for (size_t i=loup_point.is_empty()? 0 : 1; i<data.size(); i++) {
        IntervalVector box(n+1);
        if (data.is_extended_space())
            box = data[i];
        else {
            write_ext_box(data[i], box);
            box[goal_var] = Interval(uplo,loup);
            ctc.contract(box);
            if (box.is_empty()) continue;
        }

        Cell* cell=new Cell(box);
        buffer.add_property(box, cell->prop);
        bsc.add_property(box, cell->prop);
        ctc.add_property(box, cell->prop);
        loup_finder.add_property(box, cell->prop);
        buffer.push(cell);
    }

    loup_changed=false;
    initial_loup=obj_init_bound;
    time=0;

    if (cov) delete cov;
    cov = new CovOptimData(extended_COV? n+1 : n, extended_COV);
    cov->data->_optim_time = data.time();
    cov->data->_optim_nb_cells = data.nb_cells();

    // =========================================================================
    // NUEVO: NOTIFICACIÓN DE ESTADO INICIAL PARA EL AGENTE RL
    // =========================================================================
    iterations_in_macro_step = 0;
    dive_next = nullptr;
    old_gap = POS_INFINITY;
    current_macro_action = 0; 
    
    double current_gap = loup - uplo;
    
    double f1_log_nodos = std::log10(buffer.size() + 1.0);
    f1_log_nodos = f1_log_nodos / (f1_log_nodos + 2.0); 
    
    double f2_gap_pct;
    if (uplo == NEG_INFINITY || loup == POS_INFINITY) {
        f2_gap_pct = 1.0; 
    } else {
        double safe_gap = std::max(0.0, current_gap);
        double log_gap = std::log10(1.0 + safe_gap);
        f2_gap_pct = log_gap / (1.0 + log_gap); 
    }
    
    double f3_time_pct = 0.0;
    double f4_depth = 0.0;
    int f5_accion_ant = 0;
    
    std::stringstream ss;
    ss << "{ \"features\": [" 
       << f1_log_nodos << ", " << f2_gap_pct << ", " 
       << f3_time_pct << ", " << f4_depth << ", " << f5_accion_ant << "], "
       << "\"reward\": 0.0, "
       << "\"done\": false }\n";
       
    string json_response = call_python_model(ss.str());
    
    if (!json_response.empty()) {
        int decision = parse_decision_from_json(json_response);
        if (decision >= 0 && decision <= 3) {
            current_macro_action = decision;
        }
    }
    old_gap = current_gap;
}

Optimizer::Status Optimizer::optimize() {
    Timer timer;
    timer.start();

    update_uplo();

    try {
        // UNIFIED LOOP: This is the single loop that processes nodes.
        while (!buffer.empty() || dive_next != nullptr) {
            loup_changed = false;
            Cell* c = nullptr;

            // ---------------------------------------------------------
            // 1. SELECCIÓN DE NODO (NODE SELECTION)
            // ---------------------------------------------------------
            if (dive_next != nullptr) {
                // Un nodo pendiente de un dive anterior tiene prioridad SIEMPRE,
                // incluso si el agente ya cambió de heurística en esta ventana.
                // Si no lo drenamos aquí, queda huérfano (fuga de memoria) o —
                // peor — sigue manteniendo vivo el "while" de más abajo aunque
                // el buffer real esté vacío, y las ramas 0/1 intentarían hacer
                // buffer.top()/pop() sobre un buffer vacío.
                c = dive_next;
                dive_next = nullptr;
            }
            else if (current_macro_action == 2 || current_macro_action == 3) {
                // Estrategia: Feasible Diving (no hay dive pendiente, iniciamos uno)
                c = buffer.top();
                buffer.pop();
            } 
            else if (current_macro_action == 1) {
                // Estrategia: LBvUB (Araya & Neveu)
                if (static_cast<double>(rand()) / RAND_MAX < 0.5) {
                    c = buffer.top();
                    buffer.pop();
                } else {
                    std::vector<Cell*> temp_buffer;
                    double min_ub = POS_INFINITY;
                    Cell* best_ub_cell = nullptr;
                    // Búsqueda segura: limitada al tamaño real del buffer
                    int search_depth = std::min((int)buffer.size(), 100); 

                    for (int i = 0; i < search_depth; ++i) {
                        if (buffer.empty()) break; 
                        
                        Cell* curr = buffer.top();
                        buffer.pop();
                        double curr_ub = curr->box[goal_var].ub();
                        if (curr_ub < min_ub) {
                            min_ub = curr_ub;
                            if (best_ub_cell) temp_buffer.push_back(best_ub_cell);
                            best_ub_cell = curr;
                        } else {
                            temp_buffer.push_back(curr);
                        }
                    }
                    
                    if (best_ub_cell != nullptr) {
                        c = best_ub_cell;
                    } else if (!temp_buffer.empty()) {
                         c = temp_buffer.front();
                         temp_buffer.erase(temp_buffer.begin());
                    } else {
                        if (!buffer.empty()) {
                             c = buffer.top();
                             buffer.pop();
                        }
                    }

                    // Devolver los nodos evaluados a la cola
                    for (Cell* temp_cell : temp_buffer) buffer.push(temp_cell);
                }
            } 
            else {
                // Estrategia 0: Best-First (minLB)
                c = buffer.top();
                buffer.pop();
            }

            // Si c es nulo, saltamos con seguridad
            if (!c) continue;

            // [CORRECCIÓN 1] Mover iterador RL aquí para contabilizar la extracción real
            iterations_in_macro_step++;
            int current_depth = c->depth;
            g_action_counts[current_macro_action]++;

            if (trace >= 2) cout << " current box " << c->box << endl;

            // ---------------------------------------------------------
            // 2. BISECCIÓN Y EVALUACIÓN (BISECTION & EVALUATION)
            // ---------------------------------------------------------
            try {
                pair<Cell*,Cell*> new_cells = bsc.bisect(*c);
                delete c; 
                c = nullptr; 
                nb_cells += 2; 

                // ---------------------------------------------------------
                // 2b. CONTRACT & BOUND DE CADA HIJO (sin decidir destino aún)
                // ---------------------------------------------------------
                // A propósito NO usamos handle_cell() aquí: handle_cell ya
                // decide push(buffer) / delete internamente. Si lo llamamos y
                // LUEGO volvemos a tocar el mismo puntero para asignarlo a
                // dive_next o a buffer, el mismo Cell* queda insertado DOS
                // VECES en el heap (doble-push -> doble-free más adelante), o
                // si la caja quedó vacía, leemos ->box.is_empty() sobre un
                // puntero que handle_cell ya liberó (use-after-free). Por eso
                // contraemos manualmente y decidimos el destino UNA sola vez
                // por hijo: buffer, dive_next, o delete.
                Cell* child1 = new_cells.first;
                Cell* child2 = new_cells.second;

                if (child1) {
                    contract_and_bound(*child1);
                    if (child1->box.is_empty()) { delete child1; child1 = nullptr; }
                }
                if (child2) {
                    contract_and_bound(*child2);
                    if (child2->box.is_empty()) { delete child2; child2 = nullptr; }
                }

                // ---------------------------------------------------------
                // 3. GESTIÓN DE HIJOS (CHILD MANAGEMENT FOR DIVING)
                // ---------------------------------------------------------
                if (current_macro_action == 2 || current_macro_action == 3) {
                    if (child1 && child2) {
                        bool prefer_child1 = (current_macro_action == 2)
                            ? (child1->box[goal_var].lb() < child2->box[goal_var].lb())
                            : (child1->box[goal_var].ub() < child2->box[goal_var].ub());

                        dive_next = prefer_child1 ? child1 : child2;
                        buffer.push(prefer_child1 ? child2 : child1); 
                    } 
                    else if (child1) dive_next = child1;
                    else if (child2) dive_next = child2;
                } else {
                    if (child1) buffer.push(child1);
                    if (child2) buffer.push(child2);
                }

                // ---------------------------------------------------------
                // 4. LÓGICA DE PODA ORIGINAL DE IBEX (ORIGINAL PRUNING)
                // ---------------------------------------------------------
                if (uplo_of_epsboxes == NEG_INFINITY) {
                    break;
                }

                if (loup_changed) {
                    double ymax = compute_ymax();
                    buffer.contract(ymax);
                    if (ymax <= NEG_INFINITY) {
                        if (trace) cout << " infinite value for the minimum " << endl;
                        break;
                    }
                }
                
                update_uplo(); 

                if (!anticipated_upper_bounding) {
                    if (get_obj_rel_prec() < rel_eps_f || get_obj_abs_prec() < abs_eps_f)
                        break;
                }

            } 
            catch (NoBisectableVariableException& ) {
                if (c) {
                    update_uplo_of_epsboxes((c->box)[goal_var].lb());
                    delete c; 
                    c = nullptr; 
                }
                update_uplo(); 
                
                // Si la bisección falla, se salta la actualización RL en este turno para evitar cálculos inválidos
                continue; 
            }

            // ---------------------------------------------------------
            // 5. CONTROL DEL AGENTE RL (RL AGENT CONTROL)
            // ---------------------------------------------------------
            // iterations_in_macro_step ya fue incrementado en el paso 1
            if (iterations_in_macro_step >= K_ITERATIONS) {
                
                double current_gap = loup - uplo;
                double gap_reward = 0.0;

                if (old_gap < POS_INFINITY && current_gap < POS_INFINITY) {
                    double safe_old = std::max(1e-9, old_gap);
                    double safe_curr = std::max(1e-9, current_gap);
                    if (safe_old > safe_curr) {
                        gap_reward = std::log(safe_old / safe_curr);
                    }
                } else if (old_gap == POS_INFINITY && current_gap < POS_INFINITY) {
                    gap_reward = 10.0; 
                }
                
                double f1_log_nodos = std::log10(buffer.size() + 1.0);
                f1_log_nodos = f1_log_nodos / (f1_log_nodos + 2.0); 
                
                double f2_gap_pct;
                if (uplo == NEG_INFINITY || loup == POS_INFINITY) {
                    f2_gap_pct = 1.0; 
                } else {
                    double safe_gap = std::max(0.0, current_gap);
                    double log_gap = std::log10(1.0 + safe_gap);
                    f2_gap_pct = log_gap / (1.0 + log_gap); 
                }
                
                double f3_time_pct = (timeout > 0) ? std::min(1.0, timer.get_time() / timeout) : 0.0;
                
                double scale_factor = std::max(10.0, static_cast<double>(n * 2));
                double f4_depth = static_cast<double>(current_depth) / (current_depth + scale_factor);
                
                int f5_accion_ant = current_macro_action;

                std::stringstream ss;
                ss << "{ \"features\": [" 
                   << f1_log_nodos << ", " << f2_gap_pct << ", " 
                   << f3_time_pct << ", " << f4_depth << ", " << f5_accion_ant << "], "
                   << "\"reward\": " << gap_reward << ", "
                   << "\"done\": false }\n";
                
                auto call_start = std::chrono::high_resolution_clock::now();
                string json_response = call_python_model(ss.str());
                auto call_end = std::chrono::high_resolution_clock::now();
                g_python_time += std::chrono::duration<double>(call_end - call_start).count();

                old_gap = current_gap;
                iterations_in_macro_step = 0;

                if (!json_response.empty()) {
                    int decision = parse_decision_from_json(json_response);
                    if (decision >= 0 && decision <= 3) {
                        current_macro_action = decision;
                    }
                }
            }

            // Chequeo de terminación por convergencia o tiempo
            if (uplo >= loup) { status = SUCCESS; goto end; }
            if (timeout > 0) timer.check(timeout);
            time = timer.get_time();
        } 

        // =========================================================
        // FINALIZACIÓN DEL BUCLE (TERMINATION HANDLING)
        // =========================================================
        timer.stop();
        time = timer.get_time();

        if (uplo_of_epsboxes == NEG_INFINITY)
            status = UNBOUNDED_OBJ;
        else if (uplo_of_epsboxes == POS_INFINITY && (loup==POS_INFINITY || (loup==initial_loup && abs_eps_f==0 && rel_eps_f==0)))
            status = INFEASIBLE;
        else if (loup==initial_loup)
            status = NO_FEASIBLE_FOUND;
        else if (get_obj_rel_prec()>rel_eps_f && get_obj_abs_prec()>abs_eps_f)
            status = UNREACHED_PREC;
        else
            status = SUCCESS;

end:
        double final_reward = TERMINAL_BONUS_NEUTRAL;
        if (status == SUCCESS) final_reward = TERMINAL_BONUS_SUCCESS;
        else if (status == TIME_OUT) final_reward = TERMINAL_BONUS_TIMEOUT;
        else if (status == UNREACHED_PREC) final_reward = TERMINAL_BONUS_UNREACHED_PREC;
        else if (status == INFEASIBLE || status == NO_FEASIBLE_FOUND) final_reward = 1.0; 

        std::stringstream ss_end;
        ss_end << "{ \"features\": [0,0,0,0,0], \"reward\": " << final_reward << ", \"done\": true }\n";
        call_python_model(ss_end.str());
        close_model_connection();

    }
    catch (TimeOutException& ) {
        status = TIME_OUT;
        std::stringstream ss_end;
        ss_end << "{ \"features\": [0,0,0,0,0], \"reward\": " << TERMINAL_BONUS_TIMEOUT << ", \"done\": true }\n";
        call_python_model(ss_end.str());
        close_model_connection();
    }

    // [CORRECCIÓN 3] Liberar dive_next en cualquier salida posible
    if (dive_next != nullptr) {
        delete dive_next;
        dive_next = nullptr;
    }

    for (int i=0; i<(extended_COV ? n+1 : n); i++)
        cov->data->_optim_var_names.push_back(string(""));

    cov->data->_optim_optimizer_status = (unsigned int) status;
    cov->data->_optim_uplo = uplo;
    cov->data->_optim_uplo_of_epsboxes = uplo_of_epsboxes;
    cov->data->_optim_loup = loup;

    cov->data->_optim_time += time;
    cov->data->_optim_nb_cells += nb_cells;
    cov->data->_optim_loup_point = loup_point;

    IntervalVector tmp(extended_COV ? n+1 : n);

    if (extended_COV) {
        write_ext_box(loup_point, tmp);
        tmp[goal_var] = Interval(uplo,loup);
        cov->add(tmp);
    } else {
        cov->add(loup_point);
    }

    while (!buffer.empty()) {
        // [CORRECCIÓN 2] Vaciar el buffer evitando pop() devuelva void
        Cell* cell = buffer.top();
        if (extended_COV)
            cov->add(cell->box);
        else {
            read_ext_box(cell->box,tmp);
            cov->add(tmp);
        }
        buffer.pop();
        delete cell;
    }

    return status;
}

namespace {
const char* green() {
#ifndef _WIN32
    return "\033[32m";
#else
    return "";
#endif
}

const char* red(){
#ifndef _WIN32
    return "\033[31m";
#else
    return "";
#endif
}

const char* white() {
#ifndef _WIN32
    return "\033[0m";
#else
    return "";
#endif
}

}

void Optimizer::report() {

    if (!cov || !buffer.empty()) { 
        cout << " not started." << endl;
        return;
    }

    switch(status) {
    case SUCCESS:
        cout << green() << " optimization successful!" << endl;
        break;
    case INFEASIBLE:
        cout << red() << " infeasible problem" << endl;
        break;
    case NO_FEASIBLE_FOUND:
        cout << red() << " no feasible point found (the problem may be infeasible)" << endl;
        break;
    case UNBOUNDED_OBJ:
        cout << red() << " possibly unbounded objective (f*=-oo)" << endl;
        break;
    case TIME_OUT:
        cout << red() << " time limit " << timeout << "s. reached " << endl;
        break;
    case UNREACHED_PREC:
        cout << red() << " unreached precision" << endl;
        break;
    }
    cout << white() <<  endl;

    if (status==INFEASIBLE) {
        cout << " infeasible problem " << endl;
    } else {
        cout << " f* in\t[" << uplo << "," << loup << "]" << endl;
        cout << "\t(best bound)" << endl << endl;

        if (loup==initial_loup)
            cout << " x* =\t--\n\t(no feasible point found)" << endl;
        else {
            if (loup_finder.rigorous())
                cout << " x* in\t" << loup_point << endl;
            else
                cout << " x* =\t" << loup_point.lb() << endl;
            cout << "\t(best feasible point)" << endl;
        }
        cout << endl;
        double rel_prec=get_obj_rel_prec();
        double abs_prec=get_obj_abs_prec();

        cout << " relative precision on f*:\t" << rel_prec;
        if (rel_prec <= rel_eps_f)
            cout << green() << " [passed] " << white();
        cout << endl;

        cout << " absolute precision on f*:\t" << abs_prec;
        if (abs_prec <= abs_eps_f)
            cout << green() << " [passed] " << white();
        cout << endl;
    }

    cout << " cpu time used:\t\t\t" << time << "s";
    if (cov->time()!=time)
        cout << " [total=" << cov->time() << "]";
    cout << endl;
    cout << " number of cells:\t\t" << nb_cells;
    if (cov->nb_cells()!=nb_cells)
        cout << " [total=" << cov->nb_cells() << "]";
    cout << endl << endl;
    
    if (statistics) 
        cout << "  ===== Statistics ====" << endl << endl << *statistics << endl;
}

} // end namespace ibex