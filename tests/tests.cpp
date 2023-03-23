
#include "Sum.h"
#include <iostream>
#include <fstream>
#include <sstream>

#include <vector>
#include <time.h>
#include <cstdlib>
#include <random>

// Global variables
extern int SIZE_X;
extern int SIZE_Y;

using namespace std;

template <typename T>
void tttT(T &arg)
{
    arg += 1000;
    cout << "Function tttT:  " << arg << endl;
}

template <typename T>
void aaaT(T &arg)
{
    arg[0] = 0;
    cout << "Function aaaT:  " << arg << endl;
}

void lineSeparation()
{
    cout << "________________________________________________" << endl;
}

void testArrayDouble()
{
    cout << "testArrayDouble" << endl
         << endl;
    int arrayDouble[SIZE_X][SIZE_Y];
    int count = 0;
    for (int i = 0; i < SIZE_X; i++)
    {
        for (int y = 0; y < SIZE_Y; y++)
        {
            arrayDouble[i][y] = count;
            cout << arrayDouble[i][y] << "\t";
            count++;
        }
        cout << endl;
    }
    lineSeparation();
    for (int i = 0; i < SIZE_X; i++)
    {
        cout << arrayDouble[i][3] << "\t";
    }
    cout << endl;
    cout << "Vector size: " << sizeof(arrayDouble[1]) << endl;
    for (int x = 0; x < SIZE_X; x++)
    {
        cout << &arrayDouble[x] << endl;
    }
    lineSeparation();
    lineSeparation();
}

void testArrayToFunc()
{
    cout << "testArrayToFunc" << endl
         << endl;
    int arr[] = {1, 2, 3, 4, 5};
    cout << arr << endl;
    cout << &arr[1] << endl;
    cout << &arr << endl;
    cout << sizeof(arr) << endl;
    lineSeparation();
    for (int i = 0; i < sizeof(arr) / 4; i++)
    {
        cout << arr[i] << endl;
    }
    lineSeparation();
    aaaT(arr);
    for (int i = 0; i < sizeof(arr) / 4; i++)
    {
        cout << arr[i] << endl;
    }
    lineSeparation();
    lineSeparation();
}

void testOutFileFunc()
{
    cout << "testOutFileFunc" << endl
         << endl;
    int *a = new int(10);
    int *b = new int(11);
    cout << "Sum() :" << Sum(*a, *b) << endl;
    lineSeparation();
    lineSeparation();
}

void testChangeFromLink()
{
    cout << "testChangeFromLink" << endl
         << endl;
    double ss = 1231.23;
    cout << "Before  " << ss << endl;
    tttT(ss);
    cout << "After " << ss << endl;
    lineSeparation();
    lineSeparation();
}

void testRandom()
{
    cout << "testRandom" << endl
         << endl;
    for (int x = 0; x < 10; x++)
    {
        cout << rand() % 1000 << endl;
    }
    lineSeparation();
    lineSeparation();
}

void testRandomGenerator()
{
    cout << "testRandomGenerator" << endl
         << endl;
    random_device r;
    default_random_engine e1(r());
    uniform_real_distribution<float> float_dist(-10, 6);
    float mean = float_dist(e1);
    cout << "Randomly-chosen mean: " << mean << '\n';
    lineSeparation();
    lineSeparation();
}


void testVectorAsMultiArray()
{
    cout << "testVectorAsMultiArray" << endl
         << endl;

    vector<vector<float>> arr;

    int count = 0;
    for (int x = 0; x < 8; x++)
    {
        vector<float> a;
        arr.push_back(a);
        for (int y = 0; y < 8; y++)
        {
            arr[x].push_back(count);
            count++;
        }
    }
    for (int x = 0; x < 8; x++)
    {
        for (int y = 0; y < 8; y++)
        {
            cout << arr[x][y] << " ";
        }
        cout << endl;
    }
    cout << "Size " << arr.size() << endl;
    lineSeparation();
    lineSeparation();
}
void printVector(vector<vector<float>> &d_array)
{
    for (int x = 0; x < d_array.size(); x++)
    {
        for (int y = 0; y < d_array[0].size(); y++)
        {
            cout << d_array[x][y] << " ";
        }
        cout << endl;
    }
    lineSeparation();
}


int testPlus5()
{
    static int x;
    x += 5;
    return x;
}