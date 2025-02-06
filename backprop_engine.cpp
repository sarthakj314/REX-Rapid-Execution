#include "backprop_engine.h"
#include <bits/stdc++.h>
using namespace std;

/***********************************************************
 * Terminal color codes for fancy output
 ***********************************************************/
static const string RESET   = "\033[0m";
static const string RED     = "\033[31m";
static const string GREEN   = "\033[32m";
static const string YELLOW  = "\033[33m";
static const string BLUE    = "\033[34m";
static const string MAGENTA = "\033[35m";
static const string CYAN    = "\033[36m";
static const string WHITE   = "\033[37m";

/***********************************************************
 * Returns a color code for a given operation string
 ***********************************************************/
static string color_for_op(const string &op) {
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
 * Global container of all var* allocated on the heap
 ***********************************************************/
vector<var*> g_all_vars;

/***********************************************************
 * var class implementation
 ***********************************************************/

// Constructor
var::var(double val, string op, vector<var*> p, int deg)
    : _value(val), _grad(0.0), operation(move(op)), degree(deg), parents(move(p))
{ }

// Copy constructor
var::var(const var &other)
    : _value(other._value),
      _grad(0.0),
      operation(other.operation),
      degree(other.degree),
      parents(other.parents)
{ }

// Assignment: var = var
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

// Assignment: var = double
var& var::operator=(double c) {
    _value = c;
    _grad  = 0.0;
    operation.clear();
    degree  = 0;
    parents.clear();
    return *this;
}

// Accessors
double var::value() const { return _value; }
double var::grad()  const { return _grad; }

// Operator Overloads (+, -, *, /, pow)
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

// Static derivative helpers
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

        for (var* parent : node->parents) {
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

// Color-coded graph drawing
void var::draw_graph() {
    unordered_map<var*, string> labels;
    int label_count = 1;

    function<void(var*, int)> dfs = [&](var* node, int depth) {
        if (!labels.count(node)) {
            labels[node] = "Var" + to_string(label_count++);
        }
        for (int i=0; i<depth; i++) cout << "    ";

        // color
        string op_color = color_for_op(node->operation);

        // Print
        cout << "[" << labels[node] << ": " << node->_value << "]";
        if (!node->operation.empty()) {
            cout << " " << op_color << "<- " << node->operation << RESET;
        }
        cout << "\n";

        for (var* parent : node->parents) {
            dfs(parent, depth + 1);
        }
    };

    cout << "=== Computation Graph ===\n";
    dfs(this, 0);
    cout << "=========================\n";
}

/***********************************************************
 * operator<< for var
 ***********************************************************/
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

/***********************************************************
 * free_all_vars(): free the global container
 ***********************************************************/
void free_all_vars() {
    for (auto* ptr : g_all_vars) {
        delete ptr;
    }
    g_all_vars.clear();
}

/***********************************************************
 * main() for demonstration
 ***********************************************************/
/*
int main() {
    // Two stack variables
    var x(2.0), y(3.0);

    // Build an expression: z = (x * y) + sin(y)
    var z = (x * y) + y.sin();

    z.backward();

    cout << "After backward:\n";
    cout << "dz/dx= " << x.grad() << "\n";
    cout << "dz/dy= " << y.grad() << "\n\n";

    z.draw_graph();

    // Clean up all dynamic objects
    free_all_vars();

    return 0;
}
*/