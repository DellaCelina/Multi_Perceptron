#pragma once
#include "Perceptron.h"

Link::Link(Node * before, Node * after) : before(before), after(after), delta(0), deltaWeight(0) {
	weight = weight_init();
}
Link::~Link(){}

void Link::SetDeltaWeight(){
	deltaWeight += LEARNING_RATE * delta * value;
}

void Link::setValue(float value){
	(*this).value = value;
}

float Link::getValue(){
	return value;
}

