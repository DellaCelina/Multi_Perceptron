#pragma once
#include "Perceptron.h"

void printError(char* msg) {
	printf("%s", msg);
}

void MPerceptron::calculateError(){
	for (int i = 0; i < numOfOutput; i++) {
		errors[i] = 0.5 * pow(y_outputs[i] - outputs[i], 2);
		sum_error += errors[i];
	}
}


MPerceptron::MPerceptron(int numOfLayers, int * numOfNodes) :numOfLayers(numOfLayers), numOfInput(numOfNodes[0]), numOfOutput(numOfNodes[numOfLayers-1]) {
	srand(time(NULL));
	layers = new Layer*[numOfLayers * sizeof(Layer*)];
	layers[0] = new InputLayer(numOfNodes[0]);
	inputLayer = (InputLayer*)layers[0];
	for (int i = 1; i < numOfLayers-1; i++) {
		*layers[i-1] + *(layers[i] = new HiddenLayer(numOfNodes[i]));
	}
	*layers[numOfLayers-2] + *(layers[numOfLayers - 1] = new OutputLayer(numOfNodes[numOfLayers - 1]));
	outputLayer = (OutputLayer*)layers[numOfLayers - 1];

	outputs = (float*)malloc(numOfOutput * sizeof(float));
	errors = (float*)malloc(numOfOutput * sizeof(float));
}

MPerceptron::~MPerceptron(){
	delete[] layers;
	free(outputs);
	free(errors);
}

void MPerceptron::SetInput(float * inputs){
	this->inputs = inputs;
	for (int i = 0; i < numOfInput; i++) {
		INode *inode = (INode*)(*layers[0]).nodes[i];
		(*inode).SetInput(inputs[i]);
	}
}

void MPerceptron::SetYOutput(float * y_outputs){
	this->y_outputs = y_outputs;
}

float * MPerceptron::getOutput(){
	return outputs;
}

void MPerceptron::Forward(){
	for (int i = 0; i < numOfLayers; i++) {
		(*layers[i]).Output();
	}
	for (int i = 0; i < numOfOutput; i++) {
		ONode *onode = (ONode*)(*layers[numOfLayers-1]).nodes[i];
		outputs[i] = (*onode).GetOutput();
	}
}

void MPerceptron::Backward(){
	for (int i = 0; i < numOfOutput; i++) {
		ONode* node = (ONode*)(*outputLayer).nodes[i];
#if RELU == 1
#if LEAK_RELU == 1
		(*node).delta = (outputs[i] > 0) ? (y_outputs[i] - outputs[i]) : LEAK_ALPHA * (y_outputs[i] - outputs[i]);
#else
		(*node).delta = (outputs[i] > 0) ? (y_outputs[i] - outputs[i]) : 0;
#endif
#else
		(*node).delta = (y_outputs[i] - outputs[i]) * outputs[i] * (1 - outputs[i]);
#endif
	}
	for (int i = numOfLayers - 1; i >= 1; i--) {
		(*layers[i]).Back();
	}
}

void MPerceptron::AppliDeltaWeight(int numOfTraining){
	for (int i = numOfLayers - 1; i >= 1; i--) {
		(*layers[i]).Appli(numOfTraining);
	}
}

void MPerceptron::InitDeltaWeight(){
	for (int i = numOfLayers - 1; i >= 1; i--) {
		(*layers[i]).Init();
	}
}

void MPerceptron::LearnCase(float * inputs, float * outputs){
	SetInput(inputs);
	SetYOutput(outputs);
	Forward();
	calculateError();
#ifdef _DEBUG
	for (int i = 0; i < numOfOutput; i++) {
		for (int j = 0; j < (*(layers[0])).numOfNodes; j++) {
			printf("%f ", (*(((INode**)(*(layers[0])).nodes)[j])).input );
		}
		printf(": %f %f\n", this->outputs[i], y_outputs[i]);
	}
#endif
	Backward();
}

void MPerceptron::PrintCase(float * inputs){
	SetInput(inputs);
	Forward();
	for (int i = 0; i < numOfOutput; i++) {
		for (int j = 0; j < (*(layers[0])).numOfNodes; j++) {
			printf("%f ", (*(((INode**)(*(layers[0])).nodes)[j])).input );
		}
		#if IRIS == 1
		float output = this->outputs[i];
		printf(": %f ", output);
		if(output >= 0.5 && output < 1.5) printf("%s\n", "I.virginica");
		else if(output >= 1.5 && output < 2.5) printf("%s\n", "I.versicolor");
		else if(output >= 2.5 && output < 3.5) printf("%s\n", "I.setosa");
		else printf("%s\n", "Not Exist");
		#else
		printf(": %f\n", this->outputs[i]);
		#endif
	}
}

void MPerceptron::AppliTraining(int numOfTraining){
	AppliDeltaWeight(numOfTraining);
	InitDeltaWeight();
}

void MPerceptron::Learn(float * inputs, float * outputs, int tc){
	sum_error = 0;
	for (int i = 0; i < tc; i++) {
		LearnCase(inputs+i*numOfInput, outputs+i*numOfOutput);
	}
#ifdef _DEBUG
	printf("\n");
#endif
	AppliTraining(tc);
}

void MPerceptron::PrintOutput(float ** inputs, float ** outputs, int tc){
	for (int j = 0; j < tc; j++) {
		SetInput(inputs[j]);
		SetYOutput(outputs[j]);
		Forward();
		for (int i = 0; i < numOfOutput; i++) {
			printf("%f ", outputs[i]);
			printf("%f\n", y_outputs[i]);
		}
	}
}

void MPerceptron::SaveWeight(char fileName[]){
	FILE *fp = fopen(fileName, "wt");
	int i;
	fprintf(fp, "%d\n", numOfLayers);
	for (i = 0; i < numOfLayers; i++) {
		fprintf(fp, "%d ", (*(layers[i])).numOfNodes);
	}
	fprintf(fp, "\n\n");
	for (i = 1; i < numOfLayers; i++) {
		PNode** currunt = (PNode**)(*(layers[i])).nodes;
		for (int j = 0; j < (*(layers[i])).numOfNodes; j++) {
			fprintf(fp, "%f ", (*(currunt[j])).weight);
			for ((*(currunt[j])).iter = (*(currunt[j])).inputLinks.begin(); (*(currunt[j])).iter != (*(currunt[j])).inputLinks.end(); (*(currunt[j])).iter++) {
				fprintf(fp, "%f ", (**((*(currunt[j])).iter)).weight);
			}
			fprintf(fp, "\n");
		}
		fprintf(fp, "\n\n");
	}
	fclose(fp);
}

void MPerceptron::LoadWeight(char fileName[]){
	FILE *fp = fopen(fileName, "rt");
		int tmpNumOfLayers, i;
	fscanf(fp, "%d", &tmpNumOfLayers);
	if (tmpNumOfLayers != numOfLayers) throw LOADERROR;
	for (i = 0; i < numOfLayers; i++) {
		int tmpNumOfNodes;
		fscanf(fp, "%d", &tmpNumOfNodes);
		if (tmpNumOfNodes != (*(layers[i])).numOfNodes) throw LOADERROR;
	}
	for (i = 1; i < numOfLayers; i++) {
		PNode** currunt = (PNode**)(*(layers[i])).nodes;
		for (int j = 0; j < (*(layers[i])).numOfNodes; j++) {
			fscanf(fp, "%f ", &((*(currunt[j])).weight));
			for ((*(currunt[j])).iter = (*(currunt[j])).inputLinks.begin(); (*(currunt[j])).iter != (*(currunt[j])).inputLinks.end(); (*(currunt[j])).iter++) {
				fscanf(fp, "%f ", &((**((*(currunt[j])).iter)).weight));
			}
		}
	}
	
}

void MPerceptron::Test(float * inputs, int tc){
	for (int i = 0; i < tc; i++) {
		PrintCase(inputs+i*numOfInput);
	}
}
