#pragma once
#include <list>
#include <math.h>
#include <ctime>
#include <cstdlib>
#include <cstdio>

#define RELU 1
#define LEAK_RELU 1

inline float weight_init(){
	return (float)(rand() % 1000) / 1000 - 0.5;
}  

#define LAMBDA 2
#define BIAS 1
#define LEARNING_RATE 0.1
#define LEAK_ALPHA 0.01

#define IRIS 1

#define SAVE -1
#define LOADERROR 0

using namespace std;

class MPerceptron;
class Link;
class Layer;
class InputLayer;
class OutputLayer;
class HiddenLayer;
class Node;
class PNode;
class INode;
class ONode;

class MPerceptron {
private:
	float * errors;
	Layer ** layers;
	InputLayer * inputLayer;
	OutputLayer * outputLayer;
	void calculateError();
	void Forward();
	void Backward();
	void AppliDeltaWeight(int numOfTraining);
	void InitDeltaWeight();
public:
	const float * inputs;
	const float * y_outputs;
	float * outputs;
	float sum_error;

	int numOfLayers;
	int numOfInput;
	int numOfOutput;

	MPerceptron(int numOfLayers, int* numOfNodes);
	virtual ~MPerceptron();
	void SetInput(float * inputs);
	void SetYOutput(float * outputs);
	float* getOutput();
	void LearnCase(float * inputs, float * outputs);
	void AppliTraining(int numOfTraining);
	void Learn(float * inputs, float * outputs, int tc);
	void PrintOutput(float ** inputs, float ** outputs, int tc);
	void SaveWeight(char fileName[]);
	void LoadWeight(char fileName[]);
	void Test(float * inputs, int tc);
	void PrintCase(float * inputs);
};

class Layer{
	friend MPerceptron;
protected:
	int numOfNodes;
public:
	Node ** nodes;
	Layer(int n);
	virtual ~Layer();
	
	virtual Layer& operator+(Layer& layer);
	virtual void Output();
	virtual void Back();
	virtual void Init();
	virtual void Appli(int numOfTraining);
};

class InputLayer : public Layer {
public:
	InputLayer(int n);
};

class OutputLayer : public Layer {
public:
	OutputLayer(int n);
};

class HiddenLayer : public Layer {
public:
	HiddenLayer(int n);
};

class Node {
	friend Layer;
public:
	list<Link*> inputLinks;
	list<Link*> outputLinks;
	list<Link*>::iterator iter;

	Node() {};
	virtual ~Node() {};
	virtual void Output() = 0;
	virtual void Back(){};
	virtual Node& operator+(Node &node);
};

class PNode: public Node {
protected:
	inline float calculateNet();
	inline float calculateOutput(float net);
	inline void calculateDelta();
public:
	float weight;
	float deltaWeight;
	float delta;
	float output;

	PNode();
	virtual ~PNode();
	void SetDeltaWeight();
	virtual void Output();
	virtual void Back();
	virtual void Init();
	virtual void Appli(int numOfTraining);
};

class INode : public Node {	
public:
	float input;
	INode();
	virtual ~INode();
	virtual void Output();
	void SetInput(float input);
};
class ONode : public PNode {
private:

public:
	ONode();
	virtual ~ONode();
	virtual void Output();
	float GetOutput();
	virtual void Back();
};

class Link {
	friend Node;
	friend PNode;
	friend INode;
	friend ONode;

private:
	Node * before, *after;
	float value;
	float deltaWeight;
	float delta;

public:
	float weight;
	Link(Node * before, Node * after);
	virtual ~Link();
	void SetDeltaWeight();
	void setValue(float value);
	float getValue();
};
