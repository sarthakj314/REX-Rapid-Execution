#ifndef BACKPROP_ENGINE_H
#define BACKPROP_ENGINE_H

#include <iostream>      // for std::cout, std::ostream
#include <string>        // for std::string
#include <vector>        // for std::vector
#include <tuple>         // for std::tuple
#include <cmath>         // for std::sin, std::cos, etc.
#include <queue>         // for std::queue
#include <unordered_map> // for std::unordered_map
#include <functional>    // for std::function
#include <stdexcept>     // for std::invalid_argument

/***********************************************************
 * Terminal color codes for fancy output (optional)
 ***********************************************************/
static const std::string RESET   = "\033[0m";
static const std::string RED     = "\033[31m";
static const std::string GREEN   = "\033[32m";
static const std::string YELLOW  = "\033[33m";
static const std::string BLUE    = "\033[34m";
static const std::string MAGENTA = "\033[35m";
static const std::string CYAN    = "\033[36m";
static const std::string WHITE   = "\033[37m";

/***********************************************************
 * Forward declarations
 ***********************************************************/
class var;

/***********************************************************
 * Global container for all allocated var* objects.
 * Freed exactly once via free_all_vars().
 ***********************************************************/
extern std::vector<var*> g_all_vars;

/***********************************************************
 * The var Class
 ***********************************************************/
class var {
public:
    double _value;           // numeric value
    double _grad;            // gradient
    std::string operation;   // operation name, e.g. "+"
    int degree;              // for topological sort
    std::vector<var*> parents; // raw pointers to parent nodes

    // Constructors
    var(double val, std::string op = "", std::vector<var*> p = {}, int deg = 0);
    var(const var &other);

    // Assignment
    var& operator=(const var &rhs);
    var& operator=(double c);

    // Accessors
    double value() const;
    double grad()  const;

    // Arithmetic Operators
    var& operator+(var &rhs);
    var& operator+(double c);
    friend var& operator+(double c, var &rhs);

    var& operator-(var &rhs);
    var& operator-(double c);
    friend var& operator-(double c, var &rhs);

    var& operator*(var &rhs);
    var& operator*(double c);
    friend var& operator*(double c, var &rhs);

    var& operator/(var &rhs);
    var& operator/(double c);
    friend var& operator/(double c, var &rhs);

    var& pow(var &rhs);

    // Unary ops
    var& sin();
    var& cos();
    var& tan();
    var& exp();
    var& log();

    // Derivative helpers
    static std::tuple<double,double> derivative_add(var* x, var* y, double grad);
    static std::tuple<double,double> derivative_sub(var* x, var* y, double grad);
    static std::tuple<double,double> derivative_mul(var* x, var* y, double grad);
    static std::tuple<double,double> derivative_div(var* x, var* y, double grad);
    static std::tuple<double,double> derivative_pow(var* x, var* y, double grad);

    static double derivative_sin(var* x, double grad);
    static double derivative_cos(var* x, double grad);
    static double derivative_tan(var* x, double grad);
    static double derivative_exp(var* x, double grad);
    static double derivative_log(var* x, double grad);

    std::tuple<double,double> derivative(var* f, var* g, std::string &op, double &grad_val);
    double derivative(var* f, std::string &op, double &grad_val);

    // Backprop
    void backward();

    // Graph Visualization
    void draw_graph();
};

// Overloaded << for debugging
std::ostream& operator<<(std::ostream& os, const var &v);

/***********************************************************
 * free_all_vars(): Frees everything in g_all_vars
 ***********************************************************/
void free_all_vars();


//==========================================================
//         If BACKPROP_ENGINE_IMPLEMENTATION is defined,
//         compile the full definitions
//==========================================================
#ifdef BACKPROP_ENGINE_IMPLEMENTATION

/***********************************************************
 * Global container
 ***********************************************************/
std::vector<var*> g_all_vars;

/***********************************************************
 * Internal color_for_op helper
 ***********************************************************/
static std::string color_for_op(const std::string &op) {
    if      (op == "+")   return RED;
    else if (op == "-")   return YELLOW;
    else if (op == "*")   return MAGENTA;
    else if (op == "/")   return BLUE;
    else if (op == "pow") return CYAN;
    // For unary ops
    else if (op == "sin" || op == "cos" || op == "tan" ||
             op == "exp" || op == "log")
        return GREEN;
    return WHITE; // default
}

/***********************************************************
 * var Implementation
 ***********************************************************/

// Constructors
var::var(double val, std::string op, std::vector<var*> p, int deg)
    : _value(val), _grad(0.0), operation(std::move(op)), degree(deg), parents(std::move(p))
{ }

var::var(const var &other)
    : _value(other._value),
      _grad(0.0),
      operation(other.operation),
      degree(other.degree),
      parents(other.parents)
{ }

// Assignments
var& var::operator=(const var &rhs) {
    if (this != &rhs) {
        _value     = rhs._value;
        _grad      = rhs._grad;
        operation  = rhs.operation;
        degree     = rhs.degree;
        parents    = rhs.parents;
    }
    return *this;
}
var& var::operator=(double c) {
    _value = c;
    _grad  = 0.0;
    operation.clear();
    degree = 0;
    parents.clear();
    return *this;
}

// Accessors
double var::value() const { return _value; }
double var::grad()  const { return _grad; }

// Operators
var& var::operator+(var &rhs) {
    this->degree++;
    rhs.degree++;
    var* result = new var(_value + rhs._value, "+", {this, &rhs});
    g_all_vars.push_back(result);
    return *result;
}
var& var::operator+(double c) {
    this->degree++;
    var* cvar = new var(c);
    g_all_vars.push_back(cvar);
    var* result = new var(_value + c, "+", {this, cvar});
    g_all_vars.push_back(result);
    return *result;
}
var& operator+(double c, var &rhs) {
    rhs.degree++;
    var* cvar = new var(c);
    g_all_vars.push_back(cvar);
    var* result = new var(c + rhs._value, "+", {cvar, &rhs});
    g_all_vars.push_back(result);
    return *result;
}

var& var::operator-(var &rhs) {
    this->degree++;
    rhs.degree++;
    var* result = new var(_value - rhs._value, "-", {this, &rhs});
    g_all_vars.push_back(result);
    return *result;
}
var& var::operator-(double c) {
    this->degree++;
    var* cvar = new var(c);
    g_all_vars.push_back(cvar);
    var* result = new var(_value - c, "-", {this, cvar});
    g_all_vars.push_back(result);
    return *result;
}
var& operator-(double c, var &rhs) {
    rhs.degree++;
    var* cvar = new var(c);
    g_all_vars.push_back(cvar);
    var* result = new var(c - rhs._value, "-", {cvar, &rhs});
    g_all_vars.push_back(result);
    return *result;
}

var& var::operator*(var &rhs) {
    this->degree++;
    rhs.degree++;
    var* result = new var(_value * rhs._value, "*", {this, &rhs});
    g_all_vars.push_back(result);
    return *result;
}
var& var::operator*(double c) {
    this->degree++;
    var* cvar = new var(c);
    g_all_vars.push_back(cvar);
    var* result = new var(_value * c, "*", {this, cvar});
    g_all_vars.push_back(result);
    return *result;
}
var& operator*(double c, var &rhs) {
    rhs.degree++;
    var* cvar = new var(c);
    g_all_vars.push_back(cvar);
    var* result = new var(c * rhs._value, "*", {cvar, &rhs});
    g_all_vars.push_back(result);
    return *result;
}

var& var::operator/(var &rhs) {
    this->degree++;
    rhs.degree++;
    var* result = new var(_value / rhs._value, "/", {this, &rhs});
    g_all_vars.push_back(result);
    return *result;
}
var& var::operator/(double c) {
    this->degree++;
    var* cvar = new var(c);
    g_all_vars.push_back(cvar);
    var* result = new var(_value / c, "/", {this, cvar});
    g_all_vars.push_back(result);
    return *result;
}
var& operator/(double c, var &rhs) {
    rhs.degree++;
    var* cvar = new var(c);
    g_all_vars.push_back(cvar);
    var* result = new var(c / rhs._value, "/", {cvar, &rhs});
    g_all_vars.push_back(result);
    return *result;
}

var& var::pow(var &rhs) {
    this->degree++;
    rhs.degree++;
    var* result = new var(std::pow(_value, rhs._value), "pow", {this, &rhs});
    g_all_vars.push_back(result);
    return *result;
}

// Unary ops
var& var::sin() {
    this->degree++;
    var* result = new var(std::sin(_value), "sin", {this});
    g_all_vars.push_back(result);
    return *result;
}
var& var::cos() {
    this->degree++;
    var* result = new var(std::cos(_value), "cos", {this});
    g_all_vars.push_back(result);
    return *result;
}
var& var::tan() {
    this->degree++;
    var* result = new var(std::tan(_value), "tan", {this});
    g_all_vars.push_back(result);
    return *result;
}
var& var::exp() {
    this->degree++;
    var* result = new var(std::exp(_value), "exp", {this});
    g_all_vars.push_back(result);
    return *result;
}
var& var::log() {
    this->degree++;
    var* result = new var(std::log(_value), "log", {this});
    g_all_vars.push_back(result);
    return *result;
}

// Derivative helpers
tuple<double,double> var::derivative_add(var* x, var* y, double grad) {
    return {grad, grad};
}
tuple<double,double> var::derivative_sub(var* x, var* y, double grad) {
    return {grad, -grad};
}
tuple<double,double> var::derivative_mul(var* x, var* y, double grad) {
    double dx = grad * y->_value;
    double dy = grad * x->_value;
    return {dx, dy};
}
tuple<double,double> var::derivative_div(var* x, var* y, double grad) {
    double dx = grad / y->_value;
    double dy = -grad * x->_value / (y->_value * y->_value);
    return {dx, dy};
}
tuple<double,double> var::derivative_pow(var* x, var* y, double grad) {
    double dx = grad * y->_value * std::pow(x->_value, (y->_value - 1));
    double dy = grad * std::pow(x->_value, y->_value) * std::log(x->_value);
    return {dx, dy};
}
double var::derivative_sin(var* x, double grad) {
    return grad * std::cos(x->_value);
}
double var::derivative_cos(var* x, double grad) {
    return -grad * std::sin(x->_value);
}
double var::derivative_tan(var* x, double grad) {
    return grad / (std::cos(x->_value)*std::cos(x->_value));
}
double var::derivative_exp(var* x, double grad) {
    return grad * std::exp(x->_value);
}
double var::derivative_log(var* x, double grad) {
    return grad / x->_value;
}

tuple<double,double> var::derivative(var* f, var* g, string &op, double &grad_val) {
    if (op == "+") {
        return derivative_add(f, g, grad_val);
    } else if (op == "-") {
        return derivative_sub(f, g, grad_val);
    } else if (op == "*") {
        return derivative_mul(f, g, grad_val);
    } else if (op == "/") {
        return derivative_div(f, g, grad_val);
    } else if (op == "pow") {
        return derivative_pow(f, g, grad_val);
    } else {
        throw invalid_argument("Unsupported binary op: " + op);
    }
}
double var::derivative(var* f, string &op, double &grad_val) {
    if (op == "sin")  return derivative_sin(f, grad_val);
    if (op == "cos")  return derivative_cos(f, grad_val);
    if (op == "tan")  return derivative_tan(f, grad_val);
    if (op == "exp")  return derivative_exp(f, grad_val);
    if (op == "log")  return derivative_log(f, grad_val);
    throw invalid_argument("Unsupported unary op: " + op);
}

// BFS-based backprop
void var::backward() {
    unordered_map<var*, double> accum_grad;
    accum_grad[this] = 1.0;

    queue<var*> Q;
    vector<var*> topo_order;
    Q.push(this);

    unordered_map<var*, int> visits;

    // BFS => topological order
    while (!Q.empty()) {
        var* node = Q.front();
        Q.pop();
        topo_order.push_back(node);

        for (auto* parent : node->parents) {
            visits[parent]++;
            if (visits[parent] == parent->degree) {
                Q.push(parent);
            }
        }
    }

    // Accumulate
    for (auto* node : topo_order) {
        if (node->parents.empty()) continue;
        double node_grad = accum_grad[node];

        if (node->parents.size() == 1) {
            var* p = node->parents[0];
            double p_grad = derivative(p, node->operation, node_grad);
            accum_grad[p] += p_grad;
        }
        else if (node->parents.size() == 2) {
            var* p1 = node->parents[0];
            var* p2 = node->parents[1];
            auto [g1, g2] = derivative(p1, p2, node->operation, node_grad);
            accum_grad[p1] += g1;
            accum_grad[p2] += g2;
        }
    }

    // store
    for (auto & [ptr, g] : accum_grad) {
        ptr->_grad += g;
    }
}

// Graph Visualization
void var::draw_graph() {
    unordered_map<var*, string> labels;
    int label_count = 1;

    function<void(var*, int)> dfs = [&](var* node, int depth) {
        if (!labels.count(node)) {
            labels[node] = "Var" + to_string(label_count++);
        }
        for (int i=0; i<depth; i++) cout << "    ";

        string op_color = color_for_op(node->operation);
        cout << "[" << labels[node] << ": " << node->_value << "]";
        if (!node->operation.empty()) {
            cout << " " << op_color << "<- " << node->operation << RESET;
        }
        cout << "\n";

        for (auto* parent : node->parents) {
            dfs(parent, depth + 1);
        }
    };

    cout << "=== Computation Graph ===\n";
    dfs(this, 0);
    cout << "=========================\n";
}

// Overloaded <<
ostream& operator<<(ostream& os, const var &v) {
    os << "Variable\n";
    os << "Value: "     << v._value     << "\n";
    os << "Grad: "      << v._grad      << "\n";
    os << "Operation: " << v.operation  << "\n";
    os << "Degree: "    << v.degree     << "\n";
    os << "Parents: ";
    for (auto* p : v.parents) {
        os << p->_value << " ";
    }
    os << "\n\n";
    return os;
}

// free_all_vars
void free_all_vars() {
    for (auto* ptr : g_all_vars) {
        delete ptr;
    }
    g_all_vars.clear();
}

#endif // BACKPROP_ENGINE_IMPLEMENTATION