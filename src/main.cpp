#include <iostream>
#include <vector>
#include <cstdio>
#include <fstream>
#include <ostream>
#include <streambuf>
#include <ctime>

#include "../include/Matrix.hpp"
#include "../include/utils/Math.hpp"
#include "../include/NeuralNetwork.hpp"
#include "../include/mnist_reader.hpp"

using namespace std;

vector<double> mapInput(vector<uint8_t> sample)
{
    // Segundo teste:
    vector<double> input;
    for (size_t i = 0; i < sample.size(); i++)
    {
        input.push_back((double)unsigned(sample.at(i)));
    }
    return input;
}

vector<double> mapOutput(uint8_t outputSample)
{
    vector<double> target;
    for (size_t i = 0; i < 10; i++)
    {
        if (unsigned(outputSample) == i)
        {
            target.push_back(1.0);
        }
        else
        {
            target.push_back(0.0);
        }
    }
    return target;
}

int main(int argc, char **argv)
{
    // get dataset
    auto dataset = mnist::read_dataset<vector, vector, uint8_t, uint8_t>();
    // size of the data set
    cout << dataset.training_images.size() << endl; // 60000
    cout << dataset.training_labels.size() << endl;
    cout << dataset.test_images.size() << endl; // 10000
    cout << dataset.test_labels.size() << endl;

    cout << unsigned(dataset.training_labels.at(0)) << endl;       // numbers on uint8_t format
    cout << dataset.training_images.at(0).size() << endl;          // images of 28 * 28 on uint8_t format every pixel
    cout << unsigned(dataset.training_images.at(0).at(0)) << endl; // images of 28 * 28 on uint8_t format every pixel of 0 or 1 (white and black)


    double learningRate = 0.05;
    double momentum = 1;
    double bias = 1;

    vector<int> topology;
    topology.push_back(784); //input layer
    topology.push_back(32); // hidden layer
    topology.push_back(10); //output layer

    cout << "Neural network" << endl;

    NeuralNetwork *n = new NeuralNetwork(topology, 1, 2, 3, bias, learningRate, momentum);
    for (int i = 0; i < dataset.training_images.size(); i++)
    {
        vector<uint8_t> sample = dataset.training_images.at(i);
        uint8_t output = dataset.training_labels.at(i);
        vector<double> input = mapInput(sample);
        vector<double> target = mapOutput(output);
        // cout << "Training at index " << i << endl;
        n->train(input, target, bias, learningRate, momentum);
        cout << "Error: " << n->error << endl;
        cout << "Size: " << n->topologySize << endl;
        cout<<i<<endl;
    }
    return 0;
}