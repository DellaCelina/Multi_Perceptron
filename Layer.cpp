#pragma once
#include "Perceptron.h"

Layer::Layer(int n):numOfNodes(n){
}


Layer::~Layer(){
	delete[] nodes;
}

Layer& Layer::operator+(Layer& layer){
	for (int i = 0; i < numOfNodes; i++) {
		for (int j = 0; j < layer.numOfNodes; j++) {
			*nodes[i] + *layer.nodes[j];
		}
	}
	return layer;
}

void Layer::Output(){
	for (int i = 0; i < numOfNodes; i++) {
		(*nodes[i]).Output();
	}
}

InputLayer::InputLayer(int n): Layer(n) {
	nodes = new Node*[n * sizeof(INode*)];
	for (int i = 0; i< n; i++) {
		nodes[i] = new INode();
	}
}

OutputLayer::OutputLayer(int n) : Layer(n) {
	nodes = new Node*[n * sizeof(ONode*)];
	for (int i = 0; i< n; i++) {
		nodes[i] = new ONode();
	}
}

void Layer::Back(){
	for (int i = 0; i < numOfNodes; i++) {
		(*nodes[i]).Back();
	}
}

void Layer::Init(){
	for (int i = 0; i < numOfNodes; i++) {
		(*((PNode*)nodes[i])).Init();
	}
}

void Layer::Appli(int numOfTraining){
	for (int i = 0; i < numOfNodes; i++) {
		(*((PNode*)nodes[i])).Appli(numOfTraining);
	}
}

HiddenLayer::HiddenLayer(int n) : Layer(n) {
	nodes = new Node*[n * sizeof(PNode*)];
	for (int i = 0; i< n; i++) {
		nodes[i] = new PNode();
	}
}