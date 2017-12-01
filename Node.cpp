#pragma once
#include "Perceptron.h"

Node & Node::operator+(Node & node) {
	Link *tmp = new Link(this, &node);
	(*this).outputLinks.push_back(tmp);
	node.inputLinks.push_back(tmp);
	return node;
}

INode::INode() : input(0){}

INode::~INode() {};

void INode::Output() {
	for (iter = outputLinks.begin(); iter != outputLinks.end(); iter++) {
		(**iter).value = input;
	}
}

void INode::SetInput(float input){
	this->input = input;
}

PNode::PNode(): delta(0), deltaWeight(0){
	weight = weight_init();
}

PNode::~PNode(){}

void PNode::SetDeltaWeight(){
	deltaWeight += LEARNING_RATE * delta * BIAS;
}

void PNode::Output(){
	float net;
	net = calculateNet();
	output = calculateOutput(net);

	for (iter = outputLinks.begin(); iter != outputLinks.end(); iter++) {
		(**iter).value = output;
	}
}

void PNode::Back(){
	calculateDelta();
	SetDeltaWeight();
	for (iter = inputLinks.begin(); iter != inputLinks.end(); iter++) {
		(**iter).delta = delta;
		(**iter).SetDeltaWeight();
	}
}

void PNode::Init(){
	deltaWeight = 0;
	for (iter = inputLinks.begin(); iter != inputLinks.end(); iter++) {
		(**iter).deltaWeight = 0;
	}
}

void PNode::Appli(int numOfTraining){
	weight += deltaWeight / numOfTraining;
	for (iter = inputLinks.begin(); iter != inputLinks.end(); iter++) {
		(**iter).weight += (**iter).deltaWeight / numOfTraining;
	}
}

inline float PNode::calculateNet() {
	float result = BIAS * weight;
	for (iter = inputLinks.begin(); iter != inputLinks.end(); iter++) {
		result += (**iter).weight * (**iter).value;
	}
	return result;
}

inline float PNode::calculateOutput(float net){
#if RELU == 1
#if LEAK_RELU == 1
	return (net > 0) ? net : LEAK_ALPHA * net;
#else
	return (net > 0) ? net : 0;
#endif
#else
	return (float)1 / (1+exp(-LAMBDA*net));
#endif
}

inline void PNode::calculateDelta(){
	float sigma = 0;
	iter = outputLinks.begin();
#if RELU == 1
#if LEAK_RELU == 1
	delta = ((**iter).value > 0) ? 1 : LEAK_ALPHA;
#else
	delta = ((**iter).value > 0) ? 1 : 0;
#endif
#else
	delta = (**iter).value * (1 - (**iter).value);
#endif
	for (; iter != outputLinks.end(); iter++) {
		sigma += (**iter).weight * (**iter).delta;
	}
	delta *= sigma;
}

ONode::ONode() : PNode() {}

ONode::~ONode() {}

void ONode::Output() {
	float net;
	net = calculateNet();
	output = calculateOutput(net);
}

float ONode::GetOutput(){
	return output;
}

void ONode::Back(){
	SetDeltaWeight();
	for (iter = inputLinks.begin(); iter != inputLinks.end(); iter++) {
		(**iter).delta = delta;
		(**iter).SetDeltaWeight();
	}
}
