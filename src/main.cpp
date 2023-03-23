#include <iostream>
#include <fstream>
#include <sstream>

#include <vector>
#include <time.h>
#include <cstdlib>
#include <random>

#include "NueralNetController.h"
#include "NueralNetFloat.h"

using namespace std;

void testNueralNetFloat()
{
    cout << "Check NueeralNet.randoms()" << endl;

    NueralNet *net = new NueralNet();
    vector<vector<float>> arr;

    int NUERON = 3;
    int DEEPS = 2;

    int count = 0;
    for (int x = 0; x < NUERON; x++)
    {
        vector<float> a;
        arr.push_back(a);
        for (int y = 0; y < DEEPS; y++)
        {
            if (count % 2 == 0)
            {
                arr[x].push_back(count);
            }
            else
            {
                arr[x].push_back(0.0);
            }
            count++;
        }
    }

    for (int i = 0; i < NUERON; i++)
    {
        for (int y = 0; y < DEEPS; y++)
        {
            cout << arr[i][y] << "  ";
        }
        cout << endl;
    }

    net->randoms(arr);

    for (int i = 0; i < NUERON; i++)
    {
        for (int y = 0; y < DEEPS; y++)
        {
            cout << arr[i][y] << "  ";
        }
        cout << endl;
    }
}


void testNueralNetController()
{
    cout << "testNuearlNetController" << endl
         << endl;

    NueralNetController *brain = new NueralNetController(0.1F);
    brain->addLayer(6);
    brain->addLayer(10);
    brain->addLayer(4);
    brain->startNueralNetController();

    float control[] = {0, 0.3, 1, 0.1, 1, 0};
    float answer[] = {1, 0, 1, 0};

    clock_t start, end;
    start = clock();

    int count = 0;
    do
    {
        brain->setData(control, sizeof(control) / sizeof(float), 0);

        brain->learns(answer);

        if (count % 10000 == 0)
        {
            cout << "Epoch: " << count << endl;
        }
        count++;
    } while (brain->getError() > 0.0016);

    cout << "Input data" << endl;

    for (int x = 0; x < brain->layers->at(2).size(); x++)
    {
        cout << brain->layers->at(2)[x][0] << endl;
    }
    cout << "Target data" << endl;
    for (int x = 0; x < sizeof(answer) / sizeof(float); x++)
    {
        cout << answer[x] << endl;
    }

    cout << "Total iteration " << count << endl;
    cout << "Error " << brain->getError() << endl;
    end = clock();
    printf("time: %f", (end - start) / ((double)CLOCKS_PER_SEC));
    cout << endl;

    vector<string> sss = brain->saveNueralNetControllerState();
    cout << "check the weight" << endl;
    for (int x = 0; x < sss.size(); x++)
    {
        cout << sss.at(x) << endl;
    }
    // Write weight to file
    ofstream weightSave;
    weightSave.open("weightSave.w", ios_base::out);
    if (weightSave.is_open())
    {

        for (int x = 0; x < sss.size(); x++)
        {
            weightSave << sss.at(x) << endl;
        }
        weightSave.close();
    }

}
void loadStateTestNueralNetController()
{
    cout << "loadStateTestNuershow NueralNetController" << endl
         << endl;

    // Load weight from file
    ifstream weightSave;
    weightSave.open("weightSave.w", ios_base::in);
    string str;
    vector<string> vecString;

    if (weightSave.is_open())
    {

        while (weightSave)
        {
            getline(weightSave, str);
            vecString.push_back(str);
            str.clear();
        }
        weightSave.close();
    }

    // Load nueralNetController from file
    NueralNetController *brain = new NueralNetController(vecString, 0.1F);

    float control[] = {0, 0.3, 1, 0.1, 1, 0};
    float answer[] = {1, 0, 1, 0};

    clock_t start, end;
    start = clock();

    int count = 0;
    do
    {
        brain->setData(control, sizeof(control) / sizeof(float), 0);

        brain->forWards();

        if (count % 10000 == 0)
        {
            cout << "Epoch: " << count << endl;
            cout << "\r";
        }
        count++;
    } while (brain->getError() > 0.0016);

    cout << "Input data" << endl;
    for (int x = 0; x < brain->layers->at(2).size(); x++)
    {
        cout << brain->layers->at(2)[x][0] << endl;
    }

    cout << "Target data" << endl;
    for (int x = 0; x < sizeof(answer) / sizeof(float); x++)
    {
        cout << answer[x] << endl;
    }

    cout << "Total iteration " << count << endl;
    cout << "Error " << brain->getError() << endl;
    end = clock();
    printf("time: %f", (end - start) / ((double)CLOCKS_PER_SEC));
    cout << endl;

    cout << "check the weight" << endl;
    vector<string> sss = brain->saveNueralNetControllerState();

    for (int x = 0; x < sss.size(); x++)
    {
        cout << sss.at(x) << endl;
    }

}

int main(int argc, char **argv)
{
    testNueralNetFloat();
    testNueralNetController();
    loadStateTestNueralNetController();
    int t = 0;
    cin >> t;

    return 0;
}
