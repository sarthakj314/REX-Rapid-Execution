#pragma once

#include <vector>
#include <string>
#include <tuple>
#include <unordered_map>
#include <iostream>
#include <queue>
#include <functional>
#include <cmath>

using namespace std;

class var {
private:
    double value;
    double grad;
    int degree;
    string operation;
    vector<var*> parents;

public:
    // Constructor
    var(double val, string op = "", vector<var*> p = {}, int deg = 0);

    // Accessors
    double get_value();
    double get_grad();
    vector<var*> get_parents();

    // Add a parent node
    void add_child(var* child);

    // Operator overloads
    friend ostream& operator<<(ostream& os, const var& v);
    var operator=(var& other);
    var operator=(double other);

    var& operator+(var& other);
    var& operator+(double other);
    friend var& operator+(double lhs, var& rhs);

    var& operator-(var& other);
    var& operator-(double other);
    friend var& operator-(double lhs, var& rhs);

    var& operator*(var& other);
    var& operator*(double other);
    friend var& operator*(double lhs, var& rhs);

    var& operator/(var& other);
    var& operator/(double other);
    friend var& operator/(double lhs, var& rhs);

    // Unary operations
    var& sin();
    var& cos();
    var& tan();
    var& exp();
    var& log();
    var& pow(var& other);

    // Derivatives
    tuple<double, double> derivative_add(var* x, var* y, double& value);
    tuple<double, double> derivative_sub(var* x, var* y, double& value);
    tuple<double, double> derivative_mul(var* x, var* y, double& value);
    tuple<double, double> derivative_div(var* x, var* y, double& value);
    tuple<double, double> derivative_pow(var* x, var* y, double& value);

    double derivative_sin(var* x, double& value);
    double derivative_cos(var* x, double& value);
    double derivative_tan(var* x, double& value);
    double derivative_exp(var* x, double& value);
    double derivative_log(var* x, double& value);

    tuple<double, double> derivative(var* f, var* g, string& operation, double& grad);
    double derivative(var* f, string& operation, double& grad);

    // Backpropagation
    void backward();

    // Graph visualization
    void draw_graph();
};