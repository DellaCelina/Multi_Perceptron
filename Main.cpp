#pragma once
#include <cstdio>
#include <conio.h>
#include "Perceptron.h"
#define NUM_OF_INPUT 4
#define NUM_OF_OUTPUT 1
#define NUM_OF_LAYERS 4
#define TRAIN_CASE 120
#define TEST_CASE 30


float* readInput(char* fileName, int size);
float* readOutput(char* fileName, int size);
void loadFunc(MPerceptron &mp);
void checkStopFunc(MPerceptron &mp, float* inputs);

int main() {
	int numOfNodes[NUM_OF_LAYERS] = { NUM_OF_INPUT,10, 10, NUM_OF_OUTPUT };
	float* inputs = readInput("irisinput.txt", TRAIN_CASE * NUM_OF_INPUT);
	float* y_outputs = readOutput("irisoutput.txt", TRAIN_CASE * NUM_OF_OUTPUT);
	float* testInputs = readInput("irisTestInput.txt", TEST_CASE * NUM_OF_INPUT);
	
	MPerceptron mp(NUM_OF_LAYERS, numOfNodes);
	int epoch = 0;
	char ch;
	printf("Load: l, Rand: any; ");
	scanf("%c", &ch);
	if (ch == 'l') {
		try{
			loadFunc(mp);
			mp.Test(testInputs, TEST_CASE);
		}
		catch(int e){
			if(e == LOADERROR){
				printf("Load Error");
				return 0;
			}
		}
	}
	else {
		try{
		
			while (1) {
				mp.Learn(inputs, y_outputs, TRAIN_CASE);
				checkStopFunc(mp, inputs);
			}
		}
		catch(int e){
			if (e == SAVE) {
				mp.SaveWeight("Log.txt");
				printf("Save to Log.txt");
			}
		}
	}
	return 0;
}

void loadFunc(MPerceptron &mp){
	char file[256];
		printf("File Name: ");
		scanf("%s", file);
		mp.LoadWeight(file);
}

void checkStopFunc(MPerceptron &mp, float* inputs){
	static int epoch = 0;
	epoch++;
	if (kbhit() && (getch() != '\n')) {
				char ch;
				mp.Test(inputs, TRAIN_CASE);
				printf("Continue: any, Save: s, Epoch: %d, Error: %f; ", epoch, mp.sum_error);
				scanf("%c", &ch);
				if (ch == 's') throw SAVE;
			}
}

float* readInput(char * fileName, int size){
	FILE *fp = fopen(fileName, "rt");
	float *input = (float*)malloc(size*sizeof(float));
	for (int i = 0; i < size; i++) {
		fscanf(fp, "%f", input+i);
	}
	fclose(fp);
	return input;
}

float * readOutput(char * fileName, int size){
	FILE *fp = fopen(fileName, "rt");
	float *output = (float*)malloc(size * sizeof(float));
	for (int i = 0; i < size; i++) {
		fscanf(fp, "%f", output + i);
	}
	fclose(fp);
	return output;
}
