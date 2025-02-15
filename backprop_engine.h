#ifndef BACKPROP_ENGINE_H
#define BACKPROP_ENGINE_H

#include <iostream>      // for std::cout, std::ostream
#include <string>        // for std::string
#include <vector>        // for std::vector
#include <tuple>         // for std::tuple
#include <cmath>         // for std::sin, std::cos, std::exp, std::pow, std::sqrt
#include <queue>         // for std::queue
#include <unordered_map> // for std::unordered_map
#include <functional>    // for std::function
#include <stdexcept>     // for std::invalid_argument
#include <iomanip>       // for std::setw, std::setprecision
#include <algorithm>     // for std::sort, std::max, std::min

// Add using declarations for commonly used types
using std::tuple;
using std::make_tuple;
using std::vector;
using std::string;
using std::unordered_map;
using std::tie;

/***********************************************************
 * Terminal color codes for fancy output (optional)
 ***********************************************************/
extern const std::string RESET;
extern const std::string RED;
extern const std::string GREEN;
extern const std::string YELLOW;
extern const std::string BLUE;
extern const std::string MAGENTA;
extern const std::string CYAN;
extern const std::string WHITE;

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
 * Multi-Linear Map (MLM) Class Definition
 ***********************************************************/
class MLM {
private:
    std::vector<std::vector<double>> data;
    size_t rows;
    size_t cols;

public:
    // Add static debug mode
    static bool DEBUG_MODE;
    // Constructors
    MLM();
    MLM(size_t rows, size_t cols);
    MLM(const std::vector<std::vector<double>>& data);
    MLM(const MLM& other);

    // Accessors
    size_t num_rows() const;
    size_t num_cols() const;

    // Basic operations
    size_t operator[](size_t i) const;
    double& at(size_t i, size_t j);
    const double& at(size_t i, size_t j) const;

    // MLM operations
    MLM transpose() const;
    MLM matmul(const MLM& other) const;  // Matrix multiplication
    MLM hadamard(const MLM& other) const; // Element-wise multiplication
    MLM add(const MLM& other) const;
    MLM subtract(const MLM& other) const;
    
    // Scalar operations
    MLM scale(double scalar) const;
    MLM shift(double scalar) const;
    
    // Element-wise operations
    MLM apply(double (*func)(double)) const;  // For activation functions
    
    // Static constructors
    static MLM zeros(size_t rows, size_t cols);
    static MLM ones(size_t rows, size_t cols);
    static MLM identity(size_t n);
    
    // Operator overloads
    MLM& operator=(const MLM& other);
    bool operator==(const MLM& other) const;

    // Utility functions
    double sum() const;
    double max() const;
    double min() const;
    double mean() const;
    double median() const;
    double mode() const;
    double variance() const;
    double standard_deviation() const;

    // I/O
    friend std::ostream& operator<<(std::ostream& os, const MLM& m);
};

/***********************************************************
 * The var Class - Automatic Differentiation Node
 ***********************************************************/
class var {
private:
    MLM _value;           // Value at this node
    MLM _grad;            // Accumulated gradient
    std::string operation;// Operation that created this node
    int degree;          // For topological sort
    std::vector<var*> parents; // Parent nodes in computation graph
    bool is_input;      // Whether this node is an input node
    bool frozen;        // Whether this node is frozen

public:
    // Add static debug mode
    static bool DEBUG_MODE;
    // Constructors
    var(const MLM& val);
    var(const std::vector<std::vector<double>>& data);
    var(const MLM& val, std::string op, std::vector<var*> p);
    var(const var& other);

    // Assignment
    var& operator=(const var& rhs);
    var& operator=(const MLM& m);

    // set properties
    void freeze();
    void unfreeze();
    void set_input(bool is_input);

    // Accessors
    size_t num_rows() const;
    size_t num_cols() const;
    size_t num_elements() const;
    const MLM& value() const;
    const MLM& grad() const;
    const std::string& op() const;
    const std::vector<var*>& get_parents() const;
    bool get_type() const;
    int get_degree() const;
    
    // static constructors
    static var ones(size_t rows, size_t cols);
    static var zeros(size_t rows, size_t cols);
    static var identity(size_t n);

    // MLM operations
    var& matmul(var& rhs);    // Matrix multiplication
    var& transpose();          // Matrix transpose
    var& hadamard(var& rhs);   // Element-wise multiplication

    // Arithmetic Operators
    var& operator+(var& rhs);
    var& operator+(double c);
    friend var& operator+(double c, var& rhs);
    var& operator-(var& rhs);
    var& operator-(double c);
    friend var& operator-(double c, var& rhs);
    var& operator*(var& rhs);
    var& operator*(double c);  // Scalar multiplication
    friend var& operator*(double c, var& rhs);

    // Element-wise operations
    var& relu();
    var& sigmoid();
    var& tanh();

    // Derivative helpers
    static tuple<MLM, MLM> derivative_matmul(var* x, var* y, const MLM& grad);
    static tuple<MLM, MLM> derivative_hadamard(var* x, var* y, const MLM& grad);
    static tuple<MLM, MLM> derivative_add(var* x, var* y, const MLM& grad);
    static tuple<MLM, MLM> derivative_sub(var* x, var* y, const MLM& grad);
    
    static MLM derivative_scale(var* x, const MLM& grad);
    static MLM derivative_transpose(var* x, const MLM& grad);
    static MLM derivative_shift(var* x, const MLM& grad);

    static MLM derivative_relu(var* x, const MLM& grad);
    static MLM derivative_sigmoid(var* x, var* z, const MLM& grad);
    static MLM derivative_tanh(var* x, var* z, const MLM& grad);

    // Utility functions
    var& sum();
    var& mean();

    // Backpropagation
    void backward();
    void step(double learning_rate);
    void zero_grad();

    // Visualization
    void draw_graph() const;
    friend std::ostream& operator<<(std::ostream& os, const var& v);
};

#endif // BACKPROP_ENGINE_H