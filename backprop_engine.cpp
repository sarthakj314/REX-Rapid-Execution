#include "backprop_engine.h"

/***********************************************************
 * Global Variables
 ***********************************************************/
const std::string RESET   = "\033[0m";
const std::string RED     = "\033[31m";
const std::string GREEN   = "\033[32m";
const std::string YELLOW  = "\033[33m";
const std::string BLUE    = "\033[34m";
const std::string MAGENTA = "\033[35m";
const std::string CYAN    = "\033[36m";
const std::string WHITE   = "\033[37m";

// Initialize static members
bool MLM::DEBUG_MODE = false;
bool var::DEBUG_MODE = false;

/***********************************************************
 * MLM Implementation
 ***********************************************************/

// Constructors
MLM::MLM() {
    rows = 0;
    cols = 0;
}

MLM::MLM(size_t rows, size_t cols): rows(rows), cols(cols) {
    data.resize(rows, std::vector<double>(cols, 0.0));
}

MLM::MLM(const std::vector<std::vector<double>>& data): rows(data.size()), cols(data[0].size()) {
    this->data = data;
}

MLM::MLM(const MLM& other): rows(other.rows), cols(other.cols) {
    this->data = other.data;
}

// Basic operations
size_t MLM::num_rows() const {
    return this->rows;
}

size_t MLM::num_cols() const {
    return this->cols;
}

double& MLM::at(size_t i, size_t j) {
    return data[i][j];
}

const double& MLM::at(size_t i, size_t j) const {
    return data[i][j];
}

// MLM operations
MLM MLM::transpose() const {
    MLM transposed(cols, rows);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            transposed.at(j, i) = at(i, j);
        }
    }
    return transposed;
}

MLM MLM::matmul(const MLM& other) const {
    if (cols != other.rows) {
        throw std::invalid_argument("Matmul: Matrix dimensions do not match");
    }
    MLM result(rows, other.cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < other.cols; ++j) {
            for (size_t k = 0; k < cols; ++k) {
                result.at(i, j) += at(i, k) * other.at(k, j);
            }
        }
    }
    return result;
}

MLM MLM::hadamard(const MLM& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Hadamard: Matrix dimensions do not match");
    }
    MLM result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at(i, j) = at(i, j) * other.at(i, j);
        }
    }
    return result;
}

MLM MLM::add(const MLM& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Add: Matrix dimensions do not match");
    }
    MLM result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at(i, j) = at(i, j) + other.at(i, j);
        }
    }
    return result;
}

MLM MLM::subtract(const MLM& other) const {
    if (rows != other.rows || cols != other.cols) {
        throw std::invalid_argument("Subtract: Matrix dimensions do not match");
    }
    MLM result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at(i, j) = at(i, j) - other.at(i, j);
        }
    }
    return result;
}

MLM MLM::inv() const {
    if (rows != cols) {
        throw std::invalid_argument("Inverse: Matrix is not square");
    }

    std::cout << "Inv: " << *this << std::endl;

    // Create augmented matrix [A|I]
    MLM augmented(rows, 2 * cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            augmented.at(i, j) = at(i, j);
            augmented.at(i, j + cols) = (i == j) ? 1.0 : 0.0;
        }
    }

    // Gaussian elimination
    for (size_t i = 0; i < rows; i++) {
        // Find pivot
        double pivot = augmented.at(i, i);
        if (std::abs(pivot) < 1e-10) {
            throw std::runtime_error("Matrix is singular");
        }

        // Scale row i
        for (size_t j = 0; j < 2 * cols; j++) {
            augmented.at(i, j) /= pivot;
        }

        // Eliminate column i
        for (size_t j = 0; j < rows; j++) {
            if (i != j) {
                double factor = augmented.at(j, i);
                for (size_t k = 0; k < 2 * cols; k++) {
                    augmented.at(j, k) -= factor * augmented.at(i, k);
                }
            }
        }
    }

    // Extract right half
    MLM result(rows, cols);
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result.at(i, j) = augmented.at(i, j + cols);
        }
    }
    return result;
}

// Scalar operations
MLM MLM::scale(double scalar) const {
    MLM result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at(i, j) = at(i, j) * scalar;
        }
    }
    return result;
}

MLM MLM::shift(double scalar) const {
    MLM result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at(i, j) = at(i, j) + scalar;
        }
    }
    return result;
}

// Element-wise operations
MLM MLM::apply(double (*func)(double)) const {
    MLM result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at(i, j) = func(at(i, j));
        }
    }
    return result;
}

// Static constructors
MLM MLM::zeros(size_t rows, size_t cols) {
    MLM result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at(i, j) = 0.0;
        }
    }
    return result;
}

MLM MLM::ones(size_t rows, size_t cols) {
    MLM result(rows, cols);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result.at(i, j) = 1.0;
        }
    }
    return result;
}

MLM MLM::identity(size_t n) {
    MLM result(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            result.at(i, j) = (i == j) ? 1.0 : 0.0;
        }
    }
    return result;
}

double MLM::sum() const {
    double result = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result += at(i, j);
        }
    }
    return result;
}

double MLM::max() const {
    double result = at(0, 0);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result = std::max(result, at(i, j));
        }
    }
    return result;
}

double MLM::min() const {
    double result = at(0, 0);
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result = std::min(result, at(i, j));
        }
    }
    return result;
}

double MLM::mean() const {
    return sum() * (1.0 / (rows * cols));
}

double MLM::median() const {
    std::vector<double> values;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            values.push_back(at(i, j));
        }
    }
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

double MLM::mode() const {
    std::unordered_map<double, int> counts;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            counts[at(i, j)]++; 
        }
    }
    double result = at(0, 0);
    int max_count = counts[result];
    for (const auto& pair : counts) {
        if (pair.second > max_count) {
            result = pair.first;
            max_count = pair.second;
        }
    }
    return result;
}

double MLM::variance() const {
    double mean = this->mean();
    double sum_squared_diff = 0.0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            sum_squared_diff += pow(at(i, j) - mean, 2);
        }
    }
    return sum_squared_diff / (rows * cols);
}

double MLM::standard_deviation() const {
    return sqrt(variance());
}        

// Operator overloads
MLM& MLM::operator=(const MLM& other) {
    rows = other.rows;
    cols = other.cols;
    data = other.data;
    return *this;
}

bool MLM::operator==(const MLM& other) const {
    if (rows != other.rows || cols != other.cols) {
        return false;
    }
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            if (at(i, j) != other.at(i, j)) {
                return false;
            }
        }
    }
    return true;
}


// I/O
std::ostream& operator<<(std::ostream& os, const MLM& m) {
    if (MLM::DEBUG_MODE) {
        os << "[" << m.rows << "x" << m.cols << " matrix]\n";
    }
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            os << std::setw(10) << std::fixed << std::setprecision(4) << m.at(i, j) << " ";
        }
        os << std::endl;
    }
    return os;
}

// Add this implementation in the MLM class section
size_t MLM::operator[](size_t i) const {
    if (i == 0) return rows;
    if (i == 1) return cols;
    throw std::out_of_range("MLM::operator[] index out of range");
}

/***********************************************************
 * var Implementation
 ***********************************************************/

// Constructors
var::var(const MLM& val) {
    _value = val;
    operation = "";
    parents = {};
    degree = 0;
    _grad = MLM::zeros(_value.num_rows(), _value.num_cols());
    frozen = false;
}

var::var(const std::vector<std::vector<double>>& data) {
    _value = MLM(data);
    operation = "";
    parents = {};
    degree = 0;
    _grad = MLM::zeros(_value.num_rows(), _value.num_cols());
    frozen = false;
}

var::var(const MLM& val, std::string op, std::vector<var*> p) {
    _value = val;
    operation = op;
    parents = p;
    degree = 0;
    _grad = MLM::zeros(_value.num_rows(), _value.num_cols());
    frozen = false;
}

var::var(const var& other) {
    _value = other._value;
    operation = other.operation;
    parents = other.parents;
    degree = other.degree;
    _grad = other._grad;
    frozen = other.frozen;
    name = other.name;
}

// Assignment
var& var::operator=(const var& rhs) {
    if (this == &rhs) return *this;
    _value = rhs._value;
    _grad = rhs._grad;
    operation = rhs.operation;
    parents = rhs.parents;
    degree = rhs.degree;
    frozen = rhs.frozen;
    name = rhs.name;
    return *this;
}

var& var::operator=(const MLM& m) {
    _value = m;
    _grad = MLM::zeros(m[0], m[1]);
    operation = "";
    parents = {};
    degree = 0;
    frozen = false;
    return *this;
}

// set properties
void var::freeze() {
    frozen = true;
}

void var::unfreeze() {
    frozen = false;
}

void var::set_name(std::string name) {
    this->name = name;
}

void var::set_degree(int degree) {
    this->degree = degree;
}

// Accessors
size_t var::num_rows() const {
    return _value.num_rows();
}

size_t var::num_cols() const {
    return _value.num_cols();
}

size_t var::num_elements() const {
    return _value.num_rows() * _value.num_cols();
}

const MLM& var::value() const {
    return _value;
}

const MLM& var::grad() const {
    return _grad;
}

const std::string& var::op() const {
    return operation;
}

const std::vector<var*>& var::get_parents() const {
    return parents;
}

int var::get_degree() const {
    return degree;
}

// static constructors
var var::ones(size_t rows, size_t cols) {
    return var(MLM::ones(rows, cols));
}

var var::zeros(size_t rows, size_t cols) {
    return var(MLM::zeros(rows, cols));
}

var var::identity(size_t n) {
    return var(MLM::identity(n));
}

// MLM operations
var& var::matmul(var& rhs) {
    this->degree++; rhs.degree++;
    MLM result = _value.matmul(rhs._value);
    var* new_var = new var(result, "matmul", {this, &rhs});
    return *new_var;
}

var& var::transpose() {
    this->degree++;
    MLM result = _value.transpose();
    var* new_var = new var(result, "transpose", {this});
    return *new_var;
}

var& var::hadamard(var& rhs) {
    this->degree++; rhs.degree++;
    MLM result = _value.hadamard(rhs._value);
    var* new_var = new var(result, "hadamard", {this, &rhs});
    return *new_var;
}

var& var::inv() {
    this->degree++;
    MLM result = _value.inv();
    var* new_var = new var(result, "inv", {this});
    return *new_var;
}

// Arithmetic Operators
var& var::operator+(var& rhs) {
    this->degree++; rhs.degree++;
    MLM result = _value.add(rhs._value);
    var* new_var = new var(result, "add", {this, &rhs});
    return *new_var;
}

var& var::operator+(double c) {
    this->degree++;
    MLM result = _value.shift(c);
    var* new_var = new var(result, "shift", {this});
    return *new_var;
}

var& operator+(double c, var& rhs) {
    rhs.degree++;
    MLM result = rhs._value.shift(c);
    var* new_var = new var(result, "shift", {&rhs});
    return *new_var;
}

var& var::operator-(var& rhs) {
    this->degree++; rhs.degree++;
    MLM results = _value.subtract(rhs._value);
    var* new_var = new var(results, "sub", {this, &rhs});
    return *new_var;
}

var& var::operator-(double c) {
    this->degree++;
    MLM results = _value.shift(-c);
    var* new_var = new var(results, "shift", {this});
    return *new_var;
}

var& operator-(double c, var& rhs) {
    rhs.degree++;
    MLM result = rhs._value.scale(-1.).shift(c);
    var* new_var = new var(result, "shift", {&rhs});
    return *new_var;
}

var& var::operator*(var& rhs) {
    this->degree++; rhs.degree++;
    MLM results = _value.hadamard(rhs._value);
    var* new_var = new var(results, "hadamard", {this, &rhs});
    return *new_var;
}

var& var::operator*(double c) {
    MLM mult = MLM::ones(this->_value.num_rows(), this->_value.num_cols()).scale(c);
    var* tmp = new var(mult); tmp->freeze();
    return tmp->hadamard(*this);
}

var& operator*(double c, var& rhs) {
    MLM mult = MLM::ones(rhs._value.num_rows(), rhs._value.num_cols()).scale(c);
    var* tmp = new var(mult);
    return tmp->hadamard(rhs);
}

// Element-wise operations
var& var::relu() {
    this->degree++;
    auto RELU = [](double x) -> double { return x > 0 ? x : 0.0; };
    MLM result = _value.apply(RELU);
    var* new_var = new var(result, "relu", {this});
    return *new_var;
}

var& var::sigmoid() {
    this->degree++;
    auto SIGMOID = [](double x) { return 1. / (1 + exp(-x)); };
    MLM result = _value.apply(SIGMOID);
    var* new_var = new var(result, "sigmoid", {this});
    return *new_var;
}

var& var::tanh() {
    this->degree++;
    auto TANH = [](double x) { return ::tanh(x); };
    MLM result = _value.apply(TANH);
    var* new_var = new var(result, "tanh", {this});
    return *new_var;
}

// Derivative helpers
tuple<MLM, MLM> var::derivative_matmul(var* x, var* y, const MLM& grad) {
    MLM grad_x = grad.matmul(y->_value.transpose());
    MLM grad_y = x->_value.transpose().matmul(grad);
    return make_tuple(grad_x, grad_y);
}

tuple<MLM, MLM> var::derivative_hadamard(var* x, var* y, const MLM& grad) {
    MLM grad_x = grad.hadamard(y->_value);
    MLM grad_y = x->_value.hadamard(grad);
    return make_tuple(grad_x, grad_y);
}

tuple<MLM, MLM> var::derivative_add(var* x, var* y, const MLM& grad) {
    MLM grad_x = grad;
    MLM grad_y = grad;
    return make_tuple(grad_x, grad_y);
}

tuple<MLM, MLM> var::derivative_sub(var* x, var* y, const MLM& grad) {
    MLM grad_x = grad;
    MLM grad_y = grad.scale(-1.0);
    return make_tuple(grad_x, grad_y);
}

MLM var::derivative_shift(var* x, const MLM& grad) {
    MLM grad_x = grad;
    return grad_x;
}

MLM var::derivative_scale(var* x, const MLM& grad) {
    // Correct derivative: grad multiplied by the scaling factor 'c'
    // Assuming 'c' is stored or accessible via the operation metadata
    // Placeholder implementation: replace with actual scaling factor retrieval
    double c = 1.0; // TODO: Retrieve the actual scaling factor used in the forward pass
    MLM grad_x = grad.scale(c);
    return grad_x;
}

MLM var::derivative_transpose(var* x, const MLM& grad) {
    MLM grad_x = grad.transpose();
    return grad_x;
}

MLM var::derivative_relu(var* x, const MLM& grad) {
    auto RELU_DERIVATIVE = [](double val) -> double { return val > 0 ? 1.0 : 0.0; };
    MLM grad_x = x->_value.apply(RELU_DERIVATIVE);
    return grad.hadamard(grad_x);
}

MLM var::derivative_sigmoid(var* x, var* z, const MLM& grad) {
    auto SIGMOID_DERIVATIVE = [](double val) -> double { return val * (1.0 - val); };
    MLM grad_x = z->_value.apply(SIGMOID_DERIVATIVE);
    return grad.hadamard(grad_x);
}

MLM var::derivative_tanh(var* x, var* z, const MLM& grad) {
    auto TANH_DERIVATIVE = [](double val) -> double { return 1.0 - val * val; };
    MLM grad_x = z->_value.apply(TANH_DERIVATIVE);
    return grad.hadamard(grad_x);
}

// Utility functions
var& var::sum(){
    var* matmul1 = new var(MLM::ones(1, this->num_rows())); matmul1->freeze();
    var* matmul2 = new var(MLM::ones(this->num_cols(), 1)); matmul2->freeze();
    var* result = new var((*matmul1).matmul(*this).matmul(*matmul2));
    return *result;
}

var& var::mean(){
    var* result = new var(sum() * (1.0 / (_value.num_rows() * _value.num_cols())));
    return *result;
    //return sum();
}

// Backpropagation
void var::backward() {
    if (var::DEBUG_MODE) {
        std::cout << CYAN << "\n=== Starting Backward Pass ===" << RESET << "\n";
    }
    
    std::unordered_map<var*, MLM> accum_grad;
    accum_grad[this] = MLM::ones(this->_value.num_rows(), this->_value.num_cols());
    
    if (var::DEBUG_MODE) {
        std::cout << "Initialized gradient at output node:\n" << accum_grad[this] << "\n";
    }

    // Queue and vector for BFS traversal and topological ordering
    std::queue<var*> Q;
    std::vector<var*> topo_order;
    Q.push(this);

    // Track visits to each node
    std::unordered_map<var*, int> visits;

    // BFS traversal to get topological ordering
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

    // Accumulate gradients by traversing nodes in topological order
    for (auto* node : topo_order) {
        MLM node_grad = accum_grad[node];

        if (var::DEBUG_MODE) { 
            if (!node->name.empty()){
                std::cout << "node: " << node->name << ", op: " << node->operation << std::endl;
                std::cout << "node value: " << node->_value << std::endl;
                std::cout << "node grad: " << node_grad << std::endl;
            }else{
                std::cout << "op: " << node->operation << std::endl;
                std::cout << "node value: " << node->_value << std::endl;
                std::cout << "node grad: " << node_grad << std::endl;
            }
            std::cout << "--------------------------------" << std::endl;
        }


        if (node->parents.empty()) continue;

        if (node->parents.size() == 1) {
            var* p = node->parents[0];
            MLM p_grad;

            if (node->operation == "relu") {
                p_grad = derivative_relu(p, node_grad);
            }
            else if (node->operation == "sigmoid") {
                p_grad = derivative_sigmoid(p, node, node_grad);
            }
            else if (node->operation == "tanh") {
                p_grad = derivative_tanh(p, node, node_grad);
            }
            else if (node->operation == "transpose") {
                p_grad = derivative_transpose(p, node_grad);
            }
            else if (node->operation == "scale") {
                p_grad = derivative_scale(p, node_grad);
            }
            else if (node->operation == "shift") {
                p_grad = derivative_shift(p, node_grad);
            }

            if (accum_grad.find(p) == accum_grad.end()) {
                accum_grad[p] = p_grad;
            } else {
                accum_grad[p] = accum_grad[p].add(p_grad);
            }
        }
        else if (node->parents.size() == 2) {
            var* p1 = node->parents[0];
            var* p2 = node->parents[1];
            MLM g1, g2;

            if (node->operation == "matmul") {
                tie(g1, g2) = derivative_matmul(p1, p2, node_grad);
            }
            else if (node->operation == "hadamard") {
            tie(g1, g2) = derivative_hadamard(p1, p2, node_grad);
        }
        else if (node->operation == "add") {
            tie(g1, g2) = derivative_add(p1, p2, node_grad);
        }
        else if (node->operation == "sub") {
            tie(g1, g2) = derivative_sub(p1, p2, node_grad);
        }

        if (accum_grad.find(p1) == accum_grad.end()) {
            accum_grad[p1] = g1;
        } else {
            accum_grad[p1] = accum_grad[p1].add(g1);
        }

        if (accum_grad.find(p2) == accum_grad.end()) {
            accum_grad[p2] = g2;
            } else {
                accum_grad[p2] = accum_grad[p2].add(g2);
            }
        }
    }

    // Store accumulated gradients in each node
    for (auto & [ptr, grad] : accum_grad) {
        if (ptr->frozen) continue;
        ptr->_grad = ptr->_grad.add(grad);
    }
}

void var::step(double learning_rate) {
    // BFS traversal to get topological ordering
    std::queue<var*> Q;
    std::vector<var*> topo_order;

    std::unordered_map<var*, int> visits;
    Q.push(this);

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

    for (auto* node : topo_order) {
        if (node->frozen) continue;
        node->_value = node->_value.subtract(node->_grad.scale(learning_rate));
    }
}

void var::zero_grad() {
    if (DEBUG_MODE) {
        std::cout << CYAN << "\nZeroing gradients..." << RESET << std::endl;
    }

    // BFS traversal to get topological ordering
    std::queue<var*> Q;
    std::vector<var*> topo_order;
    std::unordered_map<var*, int> visits;
    Q.push(this);

    while (!Q.empty()) {
        var* node = Q.front();
        Q.pop();
        topo_order.push_back(node);

        for (var* parent : node->parents) {
            visits[parent]++;
            if (visits[parent] == parent->get_degree()) {
                Q.push(parent);
            }
        }
    }

    for (auto* node : topo_order) {
        node->_grad = MLM::zeros(node->_value.num_rows(), node->_value.num_cols());
        if (DEBUG_MODE) {
            std::cout << "Zeroing gradient for node with operation '" << node->operation 
                      << "' and shape [" << node->_value.num_rows() << "x" 
                      << node->_value.num_cols() << "]" << ", grad shape ["
                      << node->_grad.num_rows() << "x" << node->_grad.num_cols() << "]"
                      << std::endl;
        }
    }

    if (DEBUG_MODE) {
        std::cout << "Zeroed gradients for " << topo_order.size() << " nodes" << std::endl;
    }
}

// Visualization
void var::draw_graph() const {
    // First pass: BFS to label all nodes and build equation
    std::queue<const var*> label_q;
    std::unordered_map<const var*, bool> labeled;
    std::unordered_map<const var*, std::string> equation_symbols;
    int node_count = 0;
    
    label_q.push(this);
    labeled[this] = true;
    
    while (!label_q.empty()) {
        const var* current = label_q.front();
        label_q.pop();
        node_count++;
        
        // Assign name or letter symbol
        if (!current->name.empty()) {
            equation_symbols[current] = current->name;
        } else {
            char symbol = 'a' + (node_count - 1);
            equation_symbols[current] = std::string(1, symbol);
        }
        
        for (const var* parent : current->parents) {
            if (!labeled[parent]) {
                label_q.push(parent);
                labeled[parent] = true;
            }
        }
    }
    
    // Second pass: Build and print equation
    std::string equation;
    
    std::cout << CYAN << "\nComputational Graph Equation:" << RESET << "\n";
    std::cout << "===========================\n\n";
    
    // Define operator precedence
    auto get_precedence = [](const std::string& op) -> int {
        if (op == "add" || op == "sub") return 1;
        if (op == "matmul") return 2;
        if (op == "hadamard") return 3;
        if (op == "transpose" || op == "relu" || op == "sigmoid" || op == "tanh") return 4;
        return 0;
    };
    
    // Helper lambda to determine if parentheses are needed
    auto needs_parens = [&get_precedence](const var* node, const var* parent, bool is_right_operand = false) -> bool {
        if (!parent) return false;
        
        int parent_prec = get_precedence(parent->operation);
        int node_prec = get_precedence(node->operation);
        
        // Always need parens for operands of hadamard product
        if (parent->operation == "hadamard") return true;
        
        // Need parens for matrix multiplication operands if they contain add/sub
        if (parent->operation == "matmul" &&
            (node->operation == "add" || node->operation == "sub")) return true;
        
        // Need parens if parent has higher precedence
        if (node_prec < parent_prec) return true;
        
        // Need parens for right operand of subtraction if it contains add/sub
        if (is_right_operand && parent->operation == "sub" &&
            (node->operation == "add" || node->operation == "sub")) return true;
        
        return false;
    };
    
    // Recursive lambda to build equation string
    std::function<std::string(const var*)> build_equation = [&](const var* node) -> std::string {
        if (node->parents.empty()) {
            return equation_symbols[node];
        }
        
        std::string result;
        
        if (node->operation == "add") {
            result = build_equation(node->parents[0]) + " + " + build_equation(node->parents[1]);
        }
        else if (node->operation == "sub") {
            result = build_equation(node->parents[0]) + " - " + build_equation(node->parents[1]);
        }
        else if (node->operation == "matmul") {
            result = build_equation(node->parents[0]) + " @ " + build_equation(node->parents[1]);
        }
        else if (node->operation == "hadamard") {
            result = build_equation(node->parents[0]) + " ⊙ " + build_equation(node->parents[1]);
        }
        else if (node->operation == "transpose") {
            result = build_equation(node->parents[0]) + "ᵀ";
        }
        else if (node->operation == "relu") {
            result = "ReLU(" + build_equation(node->parents[0]) + ")";
        }
        else if (node->operation == "sigmoid") {
            result = "σ(" + build_equation(node->parents[0]) + ")";
        }
        else if (node->operation == "tanh") {
            result = "tanh(" + build_equation(node->parents[0]) + ")";
        }
        else if (node->operation == "scale") {
            result = "Scale(" + build_equation(node->parents[0]) + ")";
        }
        else if (node->operation == "shift") {
            result = "Shift(" + build_equation(node->parents[0]) + ")";
        }
        else {
            result = equation_symbols[node];
        }
        
        // Determine if parentheses are needed around the current result
        bool is_right_operand = false;
        if (node->operation == "sub") {
            // Check if this node is the right operand of its parent
            for (const var* parent : node->parents) {
                if (parent->operation == "sub" && parent->parents.size() == 2 && parent->parents[1] == node) {
                    is_right_operand = true;
                    break;
                }
            }
        }
        
        bool parens = needs_parens(node, node->parents.empty() ? nullptr : node->parents[0], is_right_operand);
        std::string final_result = parens ? "(" + result + ")" : result;
        
        return final_result;
    };
    
    equation = build_equation(this);
    std::cout << YELLOW << "Equation: " << RESET << equation << "\n\n";
    
    // Print variable definitions (optional, enhance as needed)
    std::cout << YELLOW << "Variables:" << RESET << "\n";
    std::queue<const var*> print_q;
    std::unordered_map<const var*, bool> visited;
    
    // Initialize queue for BFS
    print_q.push(this);
    visited[this] = true;
    
    while (!print_q.empty()) {
        const var* current = print_q.front();
        print_q.pop();
        
        // Print variable info
        std::cout << GREEN << equation_symbols[current] << RESET << ": ";
        if (!current->operation.empty()) {
            std::cout << current->operation;
        } else {
            std::cout << "weight";
        }
        std::cout << BLUE << " [" << current->value().num_rows() 
                 << "x" << current->value().num_cols() << "]" << RESET;
        if (current->frozen) {
            std::cout << RED << " (frozen)" << RESET;
        }
        std::cout << "\n";
        
        for (const var* parent : current->parents) {
            if (!visited[parent]) {
                print_q.push(parent);
                visited[parent] = true;
            }
        }
    }
    std::cout << "\n";
}

std::ostream& operator<<(std::ostream& os, const var& v) {
    if (var::DEBUG_MODE) {
        os << "[var node: " << (v.operation.empty() ? "weight" : v.operation) << "]\n";
        os << "Value:\n" << v.value();
        os << "Gradient:\n" << v.grad();
        os << "Parents: " << v.parents.size() << "\n";
        os << "Degree: " << v.get_degree() << "\n";
    } else {
        os << v.value();
    }
    return os;
}