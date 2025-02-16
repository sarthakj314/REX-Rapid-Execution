#include "backprop_engine.h"
#include <random>
#include <iostream>

#define vdd std::vector<std::vector<double>>

// Generate synthetic data
std::tuple<var, var> generate_data(int n_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0, 1);
    
    vdd x_data(n_samples, std::vector<double>(2));
    vdd y_data(n_samples, std::vector<double>(1));

    // Randomly sampling points from the plane 0.75x + 3y + 1
    for (int i = 0; i < n_samples; i++) {
        x_data[i][0] = noise(gen);
        x_data[i][1] = noise(gen);
        y_data[i][0] = 0.75 * x_data[i][0] + 3 * x_data[i][1] + 1;
    }

    var x(x_data); x.freeze(); x.set_name("x");
    var y(y_data); y.freeze(); y.set_name("y");
    return {x, y};
}

tuple<double, double, double> true_linear_regression(var& x, var& y) {
    return {0.75, 3, 1};
}

// Mean squared error loss
var& mse_loss(var& pred, var& target) {
    var* diff = new var(pred - target);
    var* sqr = new var(diff->hadamard(*diff));
    return *new var(sqr->mean());
}

int main() {
    //MLM::DEBUG_MODE = true;  // Debug matrix operations
    //var::DEBUG_MODE = true;  // Debug backprop operations
    
    // Generate synthetic data
    int n_samples = 3;
    auto [x, y] = generate_data(n_samples);
    std::cout << "x: " << x << std::endl;
    std::cout << "y: " << y << std::endl;
    
    // Create model parameters
    var W = var::ones(2, 1); W.set_name("W");
    var b = var::ones(1, 1); b.set_name("b");
    var b_broadcast = var::ones(n_samples, 1);
    b_broadcast.freeze();
    b_broadcast.set_name("B");
    
    // Training hyperparameters
    double learning_rate = 0.25;
    int epochs = 2000;
    
    std::cout << "Training linear regression...\n";
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        // Forward pass
        var pred = x.matmul(W) + b_broadcast.matmul(b);
        pred.set_name("pred");

        var loss = mse_loss(pred, y); loss.freeze();
        loss.set_name("loss");
        
        // Backward pass
        loss.zero_grad();
        loss.backward();

        /*
        if (epoch == epochs-1){
            std::cout << "Final computational graph: " << std::endl;
            loss.draw_graph();
        }
        */

        // Update parameters
        loss.step(learning_rate);

        // Print progress every few epochs
        if ((epoch + 1) % (epochs / 10) == 0) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << std::endl
                      << "\tLoss: " << loss.value().at(0, 0) << std::endl
                      << "\tW1: " << W.value().at(0, 0) << std::endl
                      << "\tW2: " << W.value().at(1, 0) << std::endl
                      << "\tb: " << b.value().at(0, 0) << "\n";
        }
        W.set_degree(0);
        b.set_degree(0);
        b_broadcast.set_degree(0);
        x.set_degree(0);
        y.set_degree(0);
    }
    
    auto [true_m1, true_m2, true_b] = true_linear_regression(x, y);
    // Print final parameters
    std::cout << "\nFinal parameters:\n";
    std::cout << "W1: " << W.value().at(0, 0) << " (true: " << true_m1 << ")\n";
    std::cout << "W2: " << W.value().at(1, 0) << " (true: " << true_m2 << ")\n";
    std::cout << "b: " << b.value().at(0, 0) << " (true: " << true_b << ")\n"; 

    return 0;
}
