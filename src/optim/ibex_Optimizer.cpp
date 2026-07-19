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

using namespace std;

const int MIN_DEPTH = 5;          
const int STAGNATION_LIMIT = 10;  

const int K_ITERATIONS = 100;    
int current_macro_action = 0;    
double old_gap = 0.0;            
int nodes_pruned_in_k_step = 0;  

const double TERMINAL_BONUS_SUCCESS         =  5.0;  
const double TERMINAL_BONUS_TIMEOUT         = -5.0;  
const double TERMINAL_BONUS_UNREACHED_PREC  = -3.0;  
const double TERMINAL_BONUS_NEUTRAL         =  0.0;  

static double g_python_time = 0.0;
static long g_wall_time = 0;
static std::chrono::time_point<std::chrono::high_resolution_clock> g_start_wall_time;
static long g_action_counts[4] = {0, 0, 0, 0};

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
    
    g_python_time = 0.0; g_wall_time = 0; g_start_wall_time = std::chrono::high_resolution_clock::now();
    for(int k=0; k<4; k++) g_action_counts[k] = 0; 
    
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
    for(int k=0; k<4; k++) g_action_counts[k] = 0;
    connect_to_model_server();
    if (cov) delete cov; cov = new CovOptimData(extended_COV? n+1 : n, extended_COV);
    cov->data->_optim_time = data.time(); cov->data->_optim_nb_cells = data.nb_cells();
}

Optimizer::Status Optimizer::optimize() {
    Timer timer;
    timer.start();

    update_uplo();

    try {
        uplo = buffer.minimum(); 
        old_gap = (loup == POS_INFINITY) ? 1e5 : (loup - uplo); 
        double initial_gap = old_gap;
        double old_uplo = uplo;
        
        nodes_pruned_in_k_step = 0;
        current_macro_action = 0; 
        
        Cell* next_node = nullptr; 
        bool alternate_fair = true; 

        while (!buffer.empty() || next_node != nullptr) {

            // [NUEVO] ¡SIN LA CONDICIÓN > 0, ENVÍA ESTADO EN EL NODO 0 Y HACE PING-PONG PERFECTO!
            if (nb_cells > 0 && nb_cells % K_ITERATIONS == 0 && next_node == nullptr) {
                uplo = buffer.minimum();
                double current_gap = (loup == POS_INFINITY) ? 1e5 : (loup - uplo);
                
                double f1_log_nodos = log10(buffer.size() + 1.0); 
                double f2_gap_pct = (initial_gap > 0) ? (current_gap / initial_gap) * 100.0 : 0.0;
                double f3_time_pct = (timeout > 0) ? (time / timeout) * 100.0 : time;
                double f4_mejora_lb = uplo - old_uplo;
                double f5_accion_ant = (double)current_macro_action;

                double alpha = 0.1;
                double reward = (old_gap - current_gap) + (alpha * nodes_pruned_in_k_step);
                
                old_gap = current_gap;
                old_uplo = uplo;
                nodes_pruned_in_k_step = 0; 
                
                std::stringstream ss;
                ss << "{ \"features\": [" 
                   << f1_log_nodos << ", " << f2_gap_pct << ", " 
                   << f3_time_pct << ", " << f4_mejora_lb << ", " << f5_accion_ant << "], "
                   << "\"reward\": " << reward << ", "
                   << "\"done\": false }\n";
                
                auto call_start = std::chrono::high_resolution_clock::now();
                string json_response = call_python_model(ss.str());
                auto call_end = std::chrono::high_resolution_clock::now();
                g_python_time += std::chrono::duration<double>(call_end - call_start).count();

                if (!json_response.empty()) {
                    int decision = parse_decision_from_json(json_response);
                    if (decision >= 0 && decision <= 3) {
                        current_macro_action = decision; 
                    }
                }
            }

            loup_changed = false;
            double old_loup_antes_de_biseccion = loup; 

            Cell *c = nullptr;

            if (next_node != nullptr) {
                c = next_node;
                next_node = nullptr;
            } 
            else if (current_macro_action == 1) {
                int batch_size = std::min((int)buffer.size(), 10);
                std::vector<Cell*> batch;
                for(int i = 0; i < batch_size; i++) {
                    batch.push_back(buffer.top());
                    buffer.pop();
                }

                int best_idx = 0;
                alternate_fair = !alternate_fair;
                
                if (alternate_fair) {
                    best_idx = 0;
                } else {
                    double min_ub = POS_INFINITY;
                    for(int i = 0; i < batch_size; i++) {
                        double current_ub = batch[i]->box[goal_var].ub();
                        if (current_ub < min_ub) {
                            min_ub = current_ub;
                            best_idx = i;
                        }
                    }
                }
                c = batch[best_idx];
                for(int i = 0; i < batch_size; i++) {
                    if (i != best_idx) buffer.push(batch[i]);
                }
            } 
            else {
                c = buffer.top();
                buffer.pop();
            }

            g_action_counts[current_macro_action]++;

            if (trace >= 2) cout << " current box " << c->box << endl;

            try {
                pair<Cell*,Cell*> new_cells = bsc.bisect(*c);
                delete c; 
                nb_cells += 2; 

                handle_cell(*new_cells.first);
                handle_cell(*new_cells.second);

                Cell* left_child = (new_cells.first->box.is_empty()) ? nullptr : new_cells.first;
                Cell* right_child = (new_cells.second->box.is_empty()) ? nullptr : new_cells.second;

                bool nuevo_ub_encontrado = (loup < old_loup_antes_de_biseccion);

                if (left_child != nullptr && right_child != nullptr) {
                    if (current_macro_action == 0 || current_macro_action == 1) {
                        buffer.push(left_child);
                        buffer.push(right_child);
                    } 
                    else if (current_macro_action == 2) {
                        if (left_child->box[goal_var].lb() < right_child->box[goal_var].lb()) {
                            next_node = left_child; buffer.push(right_child);
                        } else {
                            next_node = right_child; buffer.push(left_child);
                        }
                    } 
                    else if (current_macro_action == 3) {
                        if (nuevo_ub_encontrado) {
                            if (left_child->box.intersects(loup_point)) {
                                next_node = left_child; buffer.push(right_child);
                            } else {
                                next_node = right_child; buffer.push(left_child);
                            }
                        } else {
                            if (left_child->box[goal_var].lb() < right_child->box[goal_var].lb()) {
                                next_node = left_child; buffer.push(right_child);
                            } else {
                                next_node = right_child; buffer.push(left_child);
                            }
                        }
                    }
                } 
                else if (left_child != nullptr) {
                    if (current_macro_action == 2 || current_macro_action == 3) next_node = left_child;
                    else buffer.push(left_child);
                } 
                else if (right_child != nullptr) {
                    if (current_macro_action == 2 || current_macro_action == 3) next_node = right_child;
                    else buffer.push(right_child);
                } 
                else {
                    nodes_pruned_in_k_step++; 
                }

                if (uplo_of_epsboxes == NEG_INFINITY) break;
                if (loup_changed) {
                    double ymax = compute_ymax();
                    buffer.contract(ymax);
                    if (ymax <= NEG_INFINITY) break;
                }
                update_uplo();

                if (!anticipated_upper_bounding) 
                    if (get_obj_rel_prec()<rel_eps_f || get_obj_abs_prec()<abs_eps_f)
                        break;

                if (timeout>0) timer.check(timeout); 
                time = timer.get_time();

            }
            catch (NoBisectableVariableException& ) {
                update_uplo_of_epsboxes((c->box)[goal_var].lb());
                delete c; 
                update_uplo(); 
            }
        } // <- LLAVE RECIÉN RESTAURADA
    }
    catch (TimeOutException& ) {
        status = TIME_OUT;
    }

    double final_gap = (loup == POS_INFINITY) ? 1e5 : (loup - uplo);
    double terminal_gap_term = old_gap - final_gap;

    double terminal_status_bonus;
    switch (status) {
        case SUCCESS:          terminal_status_bonus = TERMINAL_BONUS_SUCCESS; break;
        case TIME_OUT:         terminal_status_bonus = TERMINAL_BONUS_TIMEOUT; break;
        case UNREACHED_PREC:   terminal_status_bonus = TERMINAL_BONUS_UNREACHED_PREC; break;
        default:               terminal_status_bonus = TERMINAL_BONUS_NEUTRAL; break;
    }

    double terminal_reward = terminal_gap_term + terminal_status_bonus;

    std::stringstream ss_done;
    ss_done << "{ \"features\": [0.0, 0.0, 0.0, 0.0, 0.0], \"reward\": " << terminal_reward << ", \"done\": true }\n";
    call_python_model(ss_done.str());

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
    }
    else {
        cov->add(loup_point);
    }

    while (!buffer.empty()) {
        Cell* cell = buffer.top();
        if (extended_COV)
            cov->add(cell->box);
        else {
            read_ext_box(cell->box,tmp);
            cov->add(tmp);
        }
        delete buffer.pop();
    }

    return status;
}

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

    long total_actions = g_action_counts[0] + g_action_counts[1] + g_action_counts[2] + g_action_counts[3];
    cout << endl << " === RL Node-Selection Heuristic Usage ===" << endl;
    cout << " ------------------------------------------" << endl;
    cout << "  0 (Best-First / minLB) : " << g_action_counts[0] << endl;
    cout << "  1 (LBvUB)              : " << g_action_counts[1] << endl;
    cout << "  2 (FeasibleDiving)     : " << g_action_counts[2] << endl;
    cout << "  3 (FeasibleDivingUB)   : " << g_action_counts[3] << endl;
    cout << " ------------------------------------------" << endl;
    cout << "  total node extractions : " << total_actions << endl;

    if (statistics) cout << endl << "  ===== Statistics ====" << endl << endl << *statistics << endl;
}

} // end namespace ibex