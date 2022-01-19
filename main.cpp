#include <iostream>
#include <fstream>
#include <sstream>

#include <vector>
#include <time.h>
#include <cstdlib>
#include <random>

#include "Sum.h"
#include "NueralNetFloat.h"
#include "NueralNetController.h"

using namespace std;

// Global variables
extern int SIZE_X ;
extern int SIZE_Y ;


template<typename T>
void tttT ( T& arg ) {
    arg += 1000;
    cout<<"Function tttT:  "<<arg<<endl;

}


template<typename T>
void aaaT ( T& arg ) {
    arg[0] = 0;
    cout<<"Function aaaT:  "<<arg<<endl;
}

void lineSeparation() {
    cout << "________________________________________________" << endl;

}

void testArrayDouble() {
    cout << "testArrayDouble" << endl << endl;
    int arrayDouble[SIZE_X][SIZE_Y];
    int count = 0;
    for ( int i = 0; i<SIZE_X; i++ ) {
        for ( int y = 0; y<SIZE_Y; y++ ) {
            arrayDouble[i][y] = count;
            cout << arrayDouble[i][y] << "\t";
            count++;
        }
        cout <<  endl;
    }
    lineSeparation();
    for ( int i = 0; i<SIZE_X; i++ ) {
        cout << arrayDouble[i][3] << "\t";
    }
    cout <<  endl;
    cout << "Vector size: " << sizeof ( arrayDouble[1] ) << endl;
    for ( int x = 0; x<SIZE_X; x++ ) {
        cout << &arrayDouble[x] << endl;
    }
    lineSeparation();
    lineSeparation();
}

void testArrayToFunc() {
    cout << "testArrayToFunc" << endl << endl;
    int arr[]= {1, 2, 3, 4, 5};
    cout << arr << endl;
    cout << &arr[1] << endl;
    cout << &arr << endl;
    cout << sizeof ( arr ) << endl;
    lineSeparation();
    for ( int i = 0; i<sizeof ( arr ) /4; i++ ) {
        cout << arr[i] << endl;
    }
    lineSeparation();
    aaaT ( arr );
    for ( int i = 0; i<sizeof ( arr ) /4; i++ ) {
        cout << arr[i] << endl;
    }
    lineSeparation();
    lineSeparation();
}

void testOutFileFunc() {
    cout << "testOutFileFunc" << endl << endl;
    int *a = new int ( 10 );
    int *b = new int ( 11 );
    cout << "Sum() :"  << Sum ( *a, *b )  << endl;
    lineSeparation();
    lineSeparation();
}

void testChangeFromLink() {
    cout << "testChangeFromLink" << endl << endl;
    double ss = 1231.23;
    cout << "Before  " << ss << endl;
    tttT ( ss );
    cout <<"After " <<  ss << endl;
    lineSeparation();
    lineSeparation();
}

void testRandom() {
    cout << "testRandom" << endl << endl;
    for ( int x = 0; x<10; x++ ) {
        cout << rand() % 1000<< endl;
    }
    lineSeparation();
    lineSeparation();
}

void testRandomGenerator() {
    cout << "testRandomGenerator" << endl << endl;
    random_device r;
    default_random_engine e1 ( r() );
    uniform_real_distribution<float> float_dist ( -10, 6 );
    float mean = float_dist ( e1 );
    cout << "Randomly-chosen mean: " << mean << '\n';
    lineSeparation();
    lineSeparation();
}
void testNueralNetFloat() {
    cout << "testRandomGenerator" << endl << endl;
    cout << "Check NueeralNet.randoms()" << endl ;

    NueralNet *net = new NueralNet();
    vector<vector<float>> arr;

    int NUERON = 3;
    int DEEPS = 2;

    int count = 0;
    for ( int x = 0; x<NUERON; x++ ) {
        vector<float> a;
        arr.push_back ( a );
        for ( int y = 0; y<DEEPS; y++ ) {
            if ( count % 2 == 0 ) {
                arr[x].push_back ( count );
            } else {
                arr[x].push_back ( 0.0 );
            }
            count++;
        }
    }

    for ( int i = 0; i<NUERON; i++ ) {
        for ( int y = 0; y<DEEPS; y++ ) {
            cout << arr[i][y] << "  ";
        }
        cout << endl;
    }

    net->randoms ( arr );

    for ( int i = 0; i<NUERON; i++ ) {
        for ( int y = 0; y<DEEPS; y++ ) {
            cout << arr[i][y] << "  ";
        }
        cout << endl;
    }

    lineSeparation();
    lineSeparation();
}
void testVectorAsMultiArray() {
    cout << "testVectorAsMultiArray" << endl << endl;

    vector<vector<float>> arr;

    int count = 0;
    for ( int x = 0; x<8; x++ ) {
        vector<float> a;
        arr.push_back ( a );
        for ( int y = 0; y<8; y++ ) {
            arr[x].push_back ( count );
            count++;
        }
    }
    for ( int x = 0; x<8; x++ ) {
        for ( int y = 0; y<8; y++ ) {
            cout<< arr[x][y] << " ";

        }
        cout << endl;
    }
    cout << "Size " << arr.size() << endl;
    lineSeparation();
    lineSeparation();
}
void printVector ( vector<vector<float>> &d_array ) {
    for ( int x = 0; x<d_array.size(); x++ ) {
        for ( int y = 0; y<d_array[0].size(); y++ ) {
            cout << d_array[x][y] << " ";
        }
        cout << endl;
    }
    lineSeparation();
}

void testNueralNetController() {
    cout << "testNuearlNetController" << endl << endl;

    NueralNetController *brain = new NueralNetController ( 0.1F );
    brain->addLayer ( 6 );
    brain->addLayer ( 10 );
    brain->addLayer ( 4 );
    brain->startNueralNetController();

    float control[] = {0, 0.3, 1, 0.1, 1, 0};
    float answer[] = {1, 0, 1, 0};

    clock_t start, end;
    start = clock();

    int count = 0;
    do {
        brain->setData ( control,sizeof ( control ) /sizeof ( float ), 0 );

        brain->learns ( answer );

        if ( count % 10000 == 0 ) {
            cout << "Epoch: "<< count << endl;
        }
        count++;
    } while ( brain->getError() > 0.0016 );


    cout << "Input data" << endl;

    for ( int x = 0; x < brain->layers->at ( 2 ).size(); x++ ) {
        cout << brain->layers->at ( 2 ) [x][0] << endl;
    }
    cout <<"Target data" <<  endl;
    for ( int x = 0; x < sizeof ( answer ) /sizeof ( float ); x++ ) {
        cout <<answer[x] << endl;
    }

    cout <<"Total iteration " <<  count << endl;
    cout <<"Error " <<  brain->getError() << endl;
    end = clock();
    printf ( "time: %f", ( end - start ) / ( ( double ) CLOCKS_PER_SEC ) );
    cout<< endl;


    vector<string> sss = brain->saveNueralNetControllerState();
    cout << "check the weight" << endl;
    for ( int x = 0; x<sss.size(); x++ ) {
        cout << sss.at ( x ) << endl;
    }
// Write weight to file
    ofstream weightSave;
    weightSave.open ( "weightSave.w", ios_base::out );
    if ( weightSave.is_open() ) {

        for ( int x = 0; x<sss.size(); x++ ) {
            weightSave << sss.at ( x ) << endl;
        }
        weightSave.close();
    }
    lineSeparation();
    lineSeparation();
}
void loadStateTestNueralNetController() {
    cout << "loadStateTestNuershow NueralNetController" << endl <<
    endl;

// Load weight from file
    ifstream weightSave;
    weightSave.open ( "weightSave.w", ios_base::in );
    string str;
    vector<string> vecString;

    if ( weightSave.is_open() ) {

        while ( weightSave ) {
            getline ( weightSave, str );
            vecString.push_back ( str );
            str.clear();
        }
        weightSave.close();
    }


// Load nueralNetController from file
    NueralNetController *brain = new NueralNetController ( vecString, 0.1F );

    float control[] = {0, 0.3, 1, 0.1, 1, 0};
    float answer[] = {1, 0, 1, 0};

    clock_t start, end;
    start = clock();

    int count = 0;
    do {
        brain->setData ( control,sizeof ( control ) /sizeof ( float ), 0 );

        brain->forWards();

        if ( count % 10000 == 0 ) {
            cout << "Epoch: "<< count << endl;
            cout << "\r";
        }
        count++;
    } while ( brain->getError() > 0.0016 );

    cout << "Input data" << endl;
    for ( int x = 0; x < brain->layers->at ( 2 ).size(); x++ ) {
        cout << brain->layers->at ( 2 ) [x][0] << endl;
    }

    cout <<"Target data" <<  endl;
    for ( int x = 0; x < sizeof ( answer ) /sizeof ( float ); x++ ) {
        cout <<answer[x] << endl;
    }

    cout <<"Total iteration " <<  count << endl;
    cout <<"Error " <<  brain->getError() << endl;
    end = clock();
    printf ( "time: %f", ( end - start ) / ( ( double ) CLOCKS_PER_SEC ) );
    cout<< endl;


    cout << "check the weight" << endl;
    vector<string> sss = brain->saveNueralNetControllerState();

    for ( int x = 0; x<sss.size(); x++ ) {
        cout << sss.at ( x ) << endl;
    }

    lineSeparation();
    lineSeparation();
}

int main ( int argc, char **argv ) {
    testArrayDouble();
    testArrayToFunc();
    testOutFileFunc();
    testChangeFromLink();
    testRandom();
    testRandomGenerator();
    testVectorAsMultiArray();
    testNueralNetFloat();
    testNueralNetController();
    loadStateTestNueralNetController();

    int t = 0;
    cin>> t;

    lineSeparation();
    lineSeparation();

    return 0;
}

