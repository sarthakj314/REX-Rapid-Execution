#include <bits/stdc++.h>
using namespace std;

vector<string> operations = {"+", "-", "*", "/", "pow", "sin", "cos", "tan", "exp", "log"};
const string RESET = "\033[0m";
const string RED = "\033[31m";
const string GREEN = "\033[32m";
const string YELLOW = "\033[33m";
const string BLUE = "\033[34m";
const string MAGENTA = "\033[35m";
const string CYAN = "\033[36m";

class var {
private:
    double value;
    double grad;
    int degree;
    string operation;
    vector<var*> parents;    

public:
    var(double val, string op = "", vector<var*> p = {}, int deg = 0) : value(val), operation(op), parents(p), degree(deg) {
        grad = 0;
    }

    void add_child(var* child) {
        parents.push_back(child);
    }

    double get_value() {
        return value;
    }

    double get_grad() {
        return grad;
    }

    vector<var*> get_parents() {
        return parents;
    }

    friend ostream& operator<<(ostream& os, const var& v) {
        os << "Variable" << endl;
        os << "Value: " << v.value << endl;
        os << "Grad: " << v.grad << endl;
        os << "Operation: " << v.operation << endl;
        os << "Degree: " << v.degree << endl;
        os << "Parents: ";
        for (var* parent : v.parents)
            os << parent->value << " ";
        os << endl;
        os << endl;
        return os;
    }

    var operator=(var& other) {
        this->value = other.value;
        this->grad = other.grad;
        this->operation = other.operation;
        this->parents = other.parents;
        this->degree = other.degree;
        return *this;
    }

    var operator=(double other) {
        return var(other);
    }

    var& operator+(var& other) {
        this->degree++; other.degree++;
        var* result = new var(value + other.value, "+", vector<var*>{this, &other});
        return *result;
    }

    var& operator+(double other) {
        var* other_var = new var(other);
        this->degree++; other_var->degree++;
        var* result = new var(value + other, "+", {this, other_var});
        return *result;
    }

    friend var& operator+(double lhs, var& rhs) {
        var* lhs_var = new var(lhs);
        lhs_var->degree++; rhs.degree++;
        var* result = new var(lhs + rhs.value, "+", {lhs_var, &rhs});
        return *result;
    }

    var& operator-(var& other) {
        this->degree++; other.degree++;
        var* result = new var(value - other.value, "-", vector<var*>{this, &other});
        return *result;
    }

    var& operator-(double other) {
        var* other_var = new var(other);
        this->degree++; other_var->degree++;
        var* result = new var(value - other, "-", vector<var*>{this, other_var});
        return *result;
    }

    friend var& operator-(double lhs, var& rhs) {
        var* lhs_var = new var(lhs);
        lhs_var->degree++; rhs.degree++;
        var* result = new var(lhs - rhs.value, "-", {lhs_var, &rhs});
        return *result;
    }

    var& operator*(var& other) {
        this->degree++; other.degree++;
        var* result = new var(value * other.value, "*", {this, &other});
        return *result;
    }

    var& operator*(double other) {
        var* other_var = new var(other);
        this->degree++; other_var->degree++;
        var* result = new var(value * other, "*", vector<var*>{this, other_var});
        return *result;
    }

    friend var& operator*(double lhs, var& rhs) {
        var* lhs_var = new var(lhs);
        lhs_var->degree++; rhs.degree++;
        var* result = new var(lhs * rhs.value, "*", {lhs_var, &rhs});
        return *result;
    }

    var& operator/(var& other) {
        this->degree++; other.degree++;
        var* result = new var(value / other.value, "/", {this, &other});
        return *result;
    }

    var& operator/(double other) {
        var* other_var = new var(other);
        this->degree++; other_var->degree++;
        var* result = new var(value / other, "/", vector<var*>{this, other_var});
        return *result;
    }

    friend var& operator/(double lhs, var& rhs) {
        var* lhs_var = new var(lhs);
        lhs_var->degree++; rhs.degree++;
        var* result = new var(lhs / rhs.value, "/", {lhs_var, &rhs});
        return *result;
    }

    var& sin() {
        this->degree++;
        var* result = new var(::sin(value), "sin", {this});
        return *result;
    }

    var& cos() {
        this->degree++;
        var* result = new var(::cos(value), "cos", {this});
        return *result;
    }

    var& tan() {
        this->degree++;
        var* result = new var(::tan(value), "tan", {this});
        return *result;
    }

    var& exp() {
        this->degree++;
        var* result = new var(::exp(value), "exp", {this});
        return *result;
    }

    var& log() {
        this->degree++;
        var* result = new var(::log(value), "log", {this});
        return *result;
    }

    var& pow(var& other) {
        this->degree++; other.degree++;
        var* result = new var(::pow(value, other.value), "pow", {this, &other});
        return *result;
    }

    tuple<double, double> derivative_add(var* x, var* y, double& value) {
        return {value, value};
    }

    tuple<double, double> derivative_sub(var* x, var* y, double& value) {
        return {value, -value};
    }

    tuple<double, double> derivative_mul(var* x, var* y, double& value) {
        return {value * y->value, value * x->value};
    }

    tuple<double, double> derivative_div(var* x, var* y, double& value) {
        return {value / y->value, -value * x->value / (y->value * y->value)};
    }

    tuple<double, double> derivative_pow(var* x, var* y, double& value) {
        return {value * y->value * ::pow(x->value, y->value - 1), value * ::pow(x->value, y->value) * ::log(x->value)};
    }

    double derivative_sin(var* x, double& value) {
        return value * ::cos(x->value);
    }

    double derivative_cos(var* x, double& value) {
        return -value * ::sin(x->value);
    }

    double derivative_tan(var* x, double& value) {
        return value / (::cos(x->value) * ::cos(x->value));
    }

    double derivative_exp(var* x, double& value) {
        return value * ::exp(x->value);
    }

    double derivative_log(var* x, double& value) {
        return value / x->value;
    }


    tuple<double, double> derivative(var* f, var* g, string& operation, double& grad){
        if (operation == "+") {
            return derivative_add(f, g, grad);
        } else if (operation == "-") {
            return derivative_sub(f, g, grad);
        } else if (operation == "*") {
            return derivative_mul(f, g, grad);
        } else if (operation == "/") {
            return derivative_div(f, g, grad);
        } else if (operation == "pow") {
            return derivative_pow(f, g, grad);
        } else{
            cout << "Operation not supported" << endl;
            return {0, 0};
        }
    }

    double derivative(var* f, string& operation, double& grad){
        if (operation == "sin") {
            return derivative_sin(f, grad);
        } else if (operation == "cos") {
            return derivative_cos(f, grad);
        } else if (operation == "tan") {
            return derivative_tan(f, grad);
        } else if (operation == "exp") {
            return derivative_exp(f, grad);
        } else if (operation == "log") {
            return derivative_log(f, grad);
        } else {
            cout << "Operation not supported" << endl;
            return 0;
        }
    }

    void backward() {
        // Store accumulated gradients
        unordered_map<var*, double> accumulated_gradients;
        accumulated_gradients[this] = 1.0;

        // Perform topological sort for autodiff
        queue<var*> q;
        vector<var*> topological_order;
        q.push(this);
        unordered_map<var*, int> degrees;

        while (!q.empty()) {
            var* node = q.front(); q.pop();
            topological_order.push_back(node);

            for (var* parent : node->parents) {
                degrees[parent]++;
                if (degrees[parent] == parent->degree) {
                    q.push(parent);
                }
            }
        }

        // Iterate over nodes in topological order
        for (auto node : topological_order) {
            if (node->parents.empty()) continue;

            double node_grad = accumulated_gradients[node]; // Current node's gradient

            // Process single-parent nodes
            if (node->parents.size() == 1) {
                var* parent = node->parents[0];
                double parent_grad = derivative(parent, node->operation, node_grad); // Compute gradient
                accumulated_gradients[parent] += parent_grad; // Accumulate gradient
            }

            // Process two-parent nodes
            else if (node->parents.size() == 2) {
                var* parent1 = node->parents[0];
                var* parent2 = node->parents[1];
                auto [parent_grad1, parent_grad2] = derivative(parent1, parent2, node->operation, node_grad); // Compute gradients

                accumulated_gradients[parent1] += parent_grad1;
                accumulated_gradients[parent2] += parent_grad2;
            }
        }

        // Add accumulated gradients to the true gradients
        for (auto& [node, grad] : accumulated_gradients)  node->grad += grad;
    }

    void draw_graph() {
        unordered_map<var*, string> node_labels; // Track labels for nodes
        unordered_map<var*, int> visited_count; // Count how many times a node is visited
        int label_counter = 1;

        // Helper function for recursive branch tracing
        function<void(var*, int, vector<string>&)> trace_branch = [&](var* node, int level, vector<string>& lines) {
            // Assign a label to the node if it's encountered for the first time
            if (node_labels.find(node) == node_labels.end()) {
                node_labels[node] = "Var" + to_string(label_counter++);
            }

            // Build the line for the current node
            string line = string(level * 4, ' ');

            // Display the node with its assigned label
            line += GREEN + "[" + node_labels[node] + ": " + to_string(node->get_value()) + "]" + RESET;

            // Add operation if it exists
            if (!node->operation.empty()) {
                line += " " + RED + "<- " + node->operation + RESET;
            }

            lines.push_back(line);

            // Recursively trace branches for parents
            for (var* parent : node->parents) {
                trace_branch(parent, level + 1, lines);
            }
        };

        // Start tracing from the current variable
        vector<string> lines;
        trace_branch(this, 0, lines);

        // Display the graph
        cout << MAGENTA << "=== Computation Graph ===" << RESET << endl;
        for (const string& line : lines) {
            cout << line << endl;
        }
        cout << MAGENTA << "=========================" << RESET << endl;
    }
};

int main() {
    /*
    var x = 3, y = 4;
    var z = x / y;
    z.backward();
    z.backward();
    z.draw_graph();
    cout << "dz/dx: " << x.get_grad() << endl;
    cout << "dz/dy: " << y.get_grad() << endl;
    return 0;
    */
}