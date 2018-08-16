
/**
 ==================================================
|           SCSA_2D2D Image Analysis [MKL]         |
 ==================================================
 - -* (C) Copyright 2016 King Abdullah University of Science and Technology
 Authors:
 Abderrazak Chahid (abderrazak.chahid@kaust.edu.sa)
 Taous-Meriem Laleg (taousmeriem.laleg@kaust.edu.sa)
 *
 * New function get added to code, CHAHID ROUTINE, Some variable have been changed 
 * (Class_SCSA = baboon_opts, Objet_SCSA=opts, )
 * Based on  some routines written on 2015 by :

 Ali Charara (ali.charara@kaust.edu.sa)
 David Keyes (david.keyes@kaust.edu.sa)
 Hatem Ltaief (hatem.ltaief@kaust.edu.sa)
 Taous-Meriem Laleg (taousmeriem.laleg@kaust.edu.sa)
 
 Redistribution  and  use  in  source and binary forms, with or without
 modification,  are  permitted  provided  that the following conditions
 are met:
 
 * Redistributions  of  source  code  must  retain  the above copyright
 * notice,  this  list  of  conditions  and  the  following  disclaimer.
 * Redistributions  in  binary  form must reproduce the above copyright
 * notice,  this list of conditions and the following disclaimer in the
 * documentation  and/or other materials provided with the distribution.
 * Neither  the  name of the King Abdullah University of Science and
 * chahid1/Excution_first/Code_first 
 * Technology nor the names of its contributors may be used to endorse
 * or promote products derived from this software without specific prior
 * written permission.
 *
 *
 THIS  SOFTWARE  IS  PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 ``AS IS''  AND  ANY  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 LIMITED  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 A  PARTICULAR  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 SPECIAL,  EXEMPLARY,  OR  CONSEQUENTIAL  DAMAGES  (INCLUDING,  BUT NOT
 LIMITED  TO,  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 DATA,  OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 THEORY  OF  LIABILITY,  WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 OF  THIS  SOFTWARE,  EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **/


#include <stdio.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include <vector>
#include <mkl_lapack.h>
#include <mkl_blas.h>
#include <math.h>
#include <sys/time.h>

using namespace std;

// *********************** Type DEFINITION  ***********************

// structures ==================================================================

//  Class_SCSA : it includes the needed parameter and data for SCSA algorithm

struct Class_SCSA
{
// name of needed data in .dat format
string  dataFileName;
string  deltaFileName;
string  originImage;

// matrix size
int x_dim, y_dim;

// SCSA Prameter
int d;
float h;
int gm;
float fe;

//paramters and flags
bool verbose;
unsigned long buffer_size;
char jobvl;
char jobvr;

};
// *********************** Type DEFINITION  ***********************

// function prototypes ==================================================================
// // prepare data
bool readInput(Class_SCSA* Objet_SCSA, float* &matBuffer, float* &original);
template<typename T> bool writeBuffer(Class_SCSA* Objet_SCSA, long int size, T* buffer, string fileName);

// MKL_Eigen_solver
void syevd( const char jobz, const char uplo,
           const int n,
           float* a, const int lda, float* w,
           float* work, const int lwork, int* iwork, const int liwork, int info );

// // SCSA features

bool delta(int n, float* &deltaBuffer, float fex, float feh);
int parse_Objet_SCSA( int argc, char** argv, Class_SCSA *Objet_SCSA );
double gettime(void);
inline float square(float v){return v*v;}
float ipow(float v, int p){
    float res = v;
//D    cout<<endl<<" power to =" << p<<endl;
    
    for(int q = 1; q < p; q++) res *= v;
    return res;
}


// ************************ GLOBALS   ***************************

/*========================= defines  ============================*/

//#define NUM_THREADS 40
#define sqr(a_) ((a_)*(a_))
#define PI 3.141592653589793
#define msg_disp 0            //  0: display   1: don't display
#define msg_disp_section 0
#define disp_matrices 1
#define USAGE printf("usage:\t./SCSA_2D2D_MKL  \n \
\t --data filename_noisy filename_original \n \
\t -N image_dimension\n \
\t -d value_for_d (defaults to 2)\n \
\t -h value_for_h (defaults to 0.2)\n \
\t -gm value_for_gm (defaults to 4)\n \
\t -fe value_for_fe (defaults to 1)\n \
\t [-v] verbose_messages_ON\n \
\t --help generate this help message\n \
\n \
please consult the README file for detailed instructions.\n");

        

// *********************** CHAHID FUNCTION ROUTINES   ***********************

void kron_prod (float* kprod,  float* a, int ma, int na,  float* b, int mb, int nb);
bool display_Objet_SCSA( Class_SCSA *Objet_SCSA );
void Matrx_Unit(int n,int m,float *Mtx_unit);
float SC_Sum(float h, int n,float *D1,float *D2,float *V,float *SC);
bool SCSA_2D2_full(Class_SCSA* Objet_SCSA, float* imag, float* Delta_mtrx);
void Disp_matrx(int n,int m,float *Mtx_A);
void Disp_EigenValues(int n,float *Mtx_A);
float simp3(float* f, int n, float dx);

template<class Type> bool build_SC_matrix(float hp, int N,float *D, float *V,Type * &SC_hhD,Type *max_img);
template<class Type> bool SCSA_2D2D_reconstruction(float h, int N2, int* p_Nh_size, float gm, float fe,float* matBuffer, float* Image_ref,float* SC_hhD, Type max_img, float* &V1 , double* time_Psinnor,double * time_eig, float *p_MSE0, float *p_MSE, float *p_PSNR0, float *p_PSNR);
const string currentDateTime(); 
bool save_performance(Class_SCSA* Objet_SCSA, int N, float PSNR0, float* scsa_parmter,int sz_loop);


// *********************** MAIN FUNCTION   ***********************

int main(int argc, char* argv[]){if(argc < 2){USAGE return 0; }

	if(msg_disp_section){
		cout<<" =================================================="<<endl;
		cout<<"|         SCSA_2D2D Image Analysis [MKL]           |"<<endl;
		cout<<" ================================================== "<<endl;
		cout<<endl<<endl<<"******************  Preparing Images and needed Data   ******************"<< endl;
	}

//################### Data Preparation : Load image, Deltat metrix #######################################################
 
Class_SCSA Objet_SCSA;
double comp_time0 = 0.0,time_eig=0.0,time_Psinnor=0.0, time_reconst=0.0,comp_end=0.0, data_prep=0.0, time_process=0.0 ;
       
	comp_time0= gettime();

    if(!parse_Objet_SCSA(argc, argv, &Objet_SCSA)) return -1;   
    if(msg_disp_section){cout<<endl<<"******************  SCSA process has started  ******************"<< endl;}
    if(!display_Objet_SCSA( &Objet_SCSA )) return -1;                              
    
float *matBuffer = NULL, *deltaBuffer = NULL, *Image_ref = NULL, feh = 2.0*PI / Objet_SCSA.x_dim;   

    if(!readInput(&Objet_SCSA, matBuffer,Image_ref)) return -1; 
    if(!delta(Objet_SCSA.x_dim, deltaBuffer, Objet_SCSA.fe, feh)) return -1;
    if(msg_disp){cout<<" --> Delta matrix has been generated."<<endl;}
    int Nb_loop=1;
   
     //########################## Start  of SC_hhD Generation #######################################################
    float h=Objet_SCSA.h,gm = Objet_SCSA.gm,fe = Objet_SCSA.fe, h2 = h*h,d;
    int N = Objet_SCSA.x_dim,M = Objet_SCSA.y_dim, N2 = N*M, lda = N;//, i,j,k;
     
    if(msg_disp){ if(Objet_SCSA.verbose) printf("allocating memory...\n");}
    if(msg_disp_section){
    	 if(Objet_SCSA.verbose) printf("starting first loop... Begin, please wait ... :) \n");
         cout<<" --> The SCSA process has started. Please, Wait few seconds! : "<<endl<<endl;}
   /*##############################  @ SCSA Prrocess started  ######################################################*/ 

int Nh_size, N4=N2*N2, lwork = 1 + 6*N2 + 2*N4, liwork = 3 + 5*N2;
float *SC_hhD=NULL,*V2=NULL, max_img ,MSE0, PSNR0, MSE, PSNR;
double  momry_oct = (N2*Nh_size+5*N2+N2*N2+lwork+liwork)*4;     
char name[100];//="scsa_performance";

            if(msg_disp_section){cout<<endl<<endl<<"***********************  SCSA process has started  **********************"<< endl;}
            if(!build_SC_matrix(h, N, deltaBuffer, matBuffer, SC_hhD, &max_img)) return -1;
            if(msg_disp){ cout<<endl<<" --> SC_hhD Matrix Has been generated succesfully "<<endl;}
            data_prep= gettime() - comp_time0;
            comp_time0 = gettime();
delete deltaBuffer;
	/*##############################  @ Image reconstruction ######################################################*/

            if ( ! SCSA_2D2D_reconstruction(h, N2, &Nh_size,  gm,  fe, matBuffer, Image_ref, SC_hhD, max_img, V2 , &time_Psinnor, &time_eig, &MSE0, &MSE,&PSNR0, &PSNR )) return -1;
            time_process=gettime() - comp_time0;
            if(msg_disp){cout<<endl<<" --> image max= "<< max_img << endl;}
            comp_end = time_process + data_prep ;
            time_reconst= time_process-(time_Psinnor+time_eig);

            if(msg_disp_section){
            cout<<" The computations are done with the following results :  "<< endl;
            cout<< "==> Performances:  Nh="<<Nh_size<< "    h=" << h << endl;
            cout<< "==> Totale Time  = \t\t"<<comp_end<<" sec"<< endl;
            cout<< "Data Preparation = \t\t"<< data_prep<<" sec ==>\t\t"<<100.0*data_prep/comp_end<< "%"<<endl;
            cout<< "EigenAnalysis    = \t\t"<<time_eig <<" sec ==>\t\t"<<100.0*time_eig/comp_end<< "%"<<endl;
            cout<<"EigenFunction normalization = \t"<< time_Psinnor<<" sec ==>\t\t"<<100.0*(time_Psinnor)/comp_end<< "%"<<endl;
            cout<<"Image Reconstruction time   = \t"<< time_reconst<<" sec ==>\t\t"<<100.0*time_reconst/comp_end<< "%"<<endl<<endl;
			cout<<endl<<"****************************     End of SCSA process    ***********************"<<endl;
            }

//cout<< "Single| h \t gm \t fe \t Nh \t PSNR0 \t PSNR \t MSE0 \t MSE \t Totale time \t EigenAnalysis % | "<<endl;
cout<< "\t "<< h<< "\t"<< gm<< "\t"<< fe<< "\t"<< Nh_size<< "\t"<< PSNR0<< "\t"<< PSNR<< "\t"<<MSE0<< "\t"<< MSE<< "\t"<<comp_end<< "\t"<< 100.0*time_eig/comp_end<<endl;

if(!writeBuffer(&Objet_SCSA,Objet_SCSA.x_dim*Objet_SCSA.x_dim,V2,"output_image.dat")) return -1;


    delete V2;
    delete SC_hhD;
    delete matBuffer;
    return 0;
}

// ########################## FUNCTION ROUTINES   ############################


/*============================= DELTA MATRIX   ==================================
 ____________________________________________________________________________
 | This function returns D matrix, where deltaBuffer is a second order        |
 | differentiation matrix obtained by using the Fourier pseudospectral mehtod |
 | __________________________________________________________________________ |*/

bool delta(int n, float* &deltaBuffer, float fex, float feh){
    int ex, p;//float ex[n-1];
    float dx, test_bx[n-1], test_tx[n-1], factor = feh/fex;
    factor *= factor;
    
    if(n%2 == 0){
        p = 1;
        dx = -(PI*PI)/(3.0*feh*feh)-1.0/6.0;
        for(int i = 0; i < n-1; i+=2){
            ex = n-i-1;
            p *= -1;
            test_bx[i] = test_tx[i] = -0.5 * p / square(sin( ex * feh * 0.5));
            ex = n-i-2;
            p *= -1;
            test_bx[i+1] = test_tx[i+1] = -0.5 * p / square(sin( ex * feh * 0.5));
        }
    }else{
        dx = -(PI*PI) / (3.0*feh*feh) - 1.0/12.0;
        for(int i = 0; i < n-1; i++){
            ex = n-i-1;
            test_bx[i] = -0.5 * pow(-1,  ex) * cot( ex * feh * 0.5) / (sin( ex * feh * 0.5));
            test_tx[i] = -0.5 * pow(-1, -ex) * cot(-ex * feh * 0.5) / (sin(-ex * feh * 0.5));
            
            //$            test_bx[i] = -0.5 * pow(-1,  ex) * cos( ex * feh * 0.5) / (sin( ex * feh * 0.5));
            //$            test_tx[i] = -0.5 * pow(-1, -ex) * cos(-ex * feh * 0.5) / (sin(-ex * feh * 0.5));
        }
    }
    
    unsigned long buffer_size = n * n;
    deltaBuffer = new float[ buffer_size ];
    if(!deltaBuffer)
    {
        cout << "out of memory, could not allocate "<< buffer_size <<" of memory!" << endl;
        return false;
    }
    
    int lda = n+1;
    
    for(int r = 0; r < n; r++){
        deltaBuffer[r*lda] = dx * factor;
    }
    float vL, vU;
    for(int r = 1; r < n; r++){
        vL = test_bx[n-r-1] * factor;
        vU = test_tx[n-r-1] * factor;
        
        for(int c = 0; c < n-r; c++){
            deltaBuffer[r   + c*lda] = vL;
            deltaBuffer[r*n + c*lda] = vU;
        }
    }
    return true;
}


/*================================ READ DATA    ===============================
____________________________________________________________________________
|         This function  reads the ".dat" file stored in Objet_SCSA          |
|                  to the buffer (table Pointer )  matBuffer                 |
| __________________________________________________________________________ |*/


bool readInput(Class_SCSA* Objet_SCSA, float* &matBuffer, float* &original)//, float* &deltaBuffer)
{
if(msg_disp){	if(Objet_SCSA->verbose) cout << "reading data from file: " << Objet_SCSA->dataFileName << ", data of size: " << Objet_SCSA->x_dim << "x" << Objet_SCSA->y_dim << endl;}
FILE* infile;

// load Noisy Image

if(msg_disp){cout<<endl<<"   --> Noisy Image : "<< Objet_SCSA->dataFileName.c_str();}
infile = fopen(Objet_SCSA->dataFileName.c_str(), "rb");
if(!infile)
{
cout << "could not open input file!" << endl; return 0;
}

Objet_SCSA->buffer_size = Objet_SCSA->x_dim * Objet_SCSA->y_dim;

if(msg_disp){if(Objet_SCSA->verbose) cout << "reading buffer of size: " << Objet_SCSA->buffer_size << endl;}

matBuffer = new float[ Objet_SCSA->buffer_size ];
if(!matBuffer)
{
cout << "out of memory, could not allocate "<< Objet_SCSA->buffer_size <<" of memory!" << endl;
fclose(infile);
return false;
}
unsigned long res = fread((void*)matBuffer, sizeof(float), Objet_SCSA->buffer_size, infile);
if(ferror(infile)){
cout << "error reading file!" << endl;
}
if(msg_disp_section){if(Objet_SCSA->verbose) cout << "did read " << res << " entries, content not checked though!" << endl;}
fclose(infile);

// load Original Image 

if(msg_disp){cout<<endl<<"   --> Reference Image :  "<< Objet_SCSA->originImage.c_str()<<endl;}
infile = fopen(Objet_SCSA->originImage.c_str(), "rb");
if(!infile)
{
cout << "could not open input file!" << endl; return 0;
}

Objet_SCSA->buffer_size = Objet_SCSA->x_dim * Objet_SCSA->y_dim;

if(Objet_SCSA->verbose) cout << "reading buffer of size: " << Objet_SCSA->buffer_size << endl;
original = new float[ Objet_SCSA->buffer_size ];
if(!original)
{
cout << "out of memory, could not allocate "<< Objet_SCSA->buffer_size <<" of memory!" << endl;
fclose(infile);
return false;
}
unsigned long res2 = fread((void*)original, sizeof(float), Objet_SCSA->buffer_size, infile);
if(ferror(infile)){
cout << "error reading file!" << endl;
}
if(Objet_SCSA->verbose) cout << "did read " << res2 << " entries, content not checked though!" << endl;
fclose(infile);


if(msg_disp){cout<<endl<<" --> The input images has been read succesfully and stored in matBuffer, Image_ref.."<<endl;}

return true;
}


/*========================= WRITE results  DATA    ===========================
____________________________________________________________________________
|       This function writes the the buffer (table Pointer )  buffer         |
|                   to  "fileName.dat" of Objet_SCSA                         |
| __________________________________________________________________________ |*/

template<typename T> bool writeBuffer(Class_SCSA* Objet_SCSA, long int size, T* buffer, string fileName)
{
	if(Objet_SCSA->verbose) printf("allocating memory...\n");
	if(Objet_SCSA->verbose) cout << "writing data to file: " << fileName << ", data of size: " << size << endl;
FILE* outfile;
//load matrix data
outfile = fopen(fileName.c_str(), "wb");
if(!outfile)
{
cout << "could not open output file!" << endl; return 0;
}

unsigned long res = fwrite((void*)buffer, sizeof(T), size, outfile);
if(ferror(outfile)){
cout << "error writing to file!" << endl;
return 0;
}
fclose(outfile);
if(msg_disp_section){
	cout<<endl<< " --> Saving data : " << endl;
	cout<<"   --> "<<fileName<<" has been created succefully."<<endl;
}

return true;
}

/*===================== SCSA Object reconstruction     =======================
____________________________________________________________________________
| This function contains the differents information of about the input data  |
| to process, Moreover it stores also SCSA parameters to use in the Process  |
|              All in type structure object called  Objet_SCSA               |
| __________________________________________________________________________ |*/


int parse_Objet_SCSA( int argc, char** argv, Class_SCSA *Objet_SCSA )
{
// fill in default values
Objet_SCSA->x_dim = 64;
Objet_SCSA->y_dim = 64;

Objet_SCSA->d = 2;
Objet_SCSA->h = 0.245;
Objet_SCSA->gm = 1;
Objet_SCSA->fe = 1.0;
//Objet_SCSA->L = 0.015915494309190;
Objet_SCSA->dataFileName = "scsa_input64.dat";
Objet_SCSA->deltaFileName = "baboon_D.dat";
Objet_SCSA->originImage = "scsa_original64.dat"; ;
Objet_SCSA->jobvl = 'N';
Objet_SCSA->jobvr = 'V';
Objet_SCSA->verbose= false;

int info;
for( int i = 1; i < argc; ++i ) {
// ----- matrix size
// each -N fills in next entry of size
if ( strcmp("-N", argv[i]) == 0 && i+1 < argc ) {
    i++;
    int m, n;
    info = sscanf( argv[i], "%d:%d", &m, &n);
    if ( info == 2 && m > 0 && n > 0 ) {
        Objet_SCSA->x_dim = m;
        Objet_SCSA->y_dim = n;
    }
    else if ( info == 1 && m >= 0 ) {
        Objet_SCSA->x_dim = m;
        Objet_SCSA->y_dim = m;// implicitly
    }
    else {
        fprintf( stderr, "error: -N %s is invalid; ensure m >= 0, n >= 0, info=%d, m=%d, n=%d.\n",
                argv[i],info,m,n);
        exit(1);
    }
}


else if ( strcmp("--data", argv[i]) == 0 && i+1 < argc ) {
    Objet_SCSA->dataFileName = argv[++i];
    Objet_SCSA->originImage = argv[++i];
}
else if ( strcmp("--delta", argv[i]) == 0 && i+1 < argc ) {
    Objet_SCSA->deltaFileName = argv[++i];
}
else if ( strcmp("-d",    argv[i]) == 0 && i+1 < argc ) {
    Objet_SCSA->d = atoi( argv[++i] );
}
else if ( strcmp("-h", argv[i]) == 0 && i+1 < argc ) {
    Objet_SCSA->h = atof( argv[++i] );
}
else if ( strcmp("-fe", argv[i]) == 0 && i+1 < argc ) {
    Objet_SCSA->fe = atof( argv[++i] );
}
else if ( strcmp("-gm",   argv[i]) == 0 && i+1 < argc ) {
    Objet_SCSA->gm = atoi( argv[++i] );
    //Objet_SCSA->L = 1.0 / (4*PI*(Objet_SCSA->gm+1));
}/*
  else if ( strcmp("-L", argv[i]) == 0 && i+1 < argc ) {
  Objet_SCSA->L = atof( argv[++i] );
  }*/
else if ( strcmp("-v",  argv[i]) == 0 ) { Objet_SCSA->verbose= true;  }

// ----- usage
else if ( strcmp("--help", argv[i]) == 0 ) {
    USAGE
    exit(0);
}
else {
    fprintf( stderr, "error: unrecognized option %s\n", argv[i] );
    exit(1);
}
}

return 1;
}


/*========================== Time measurement     ============================
 ____________________________________________________________________________
 |       This function returns the actual time in secondes                   |
 | __________________________________________________________________________ |*/

double gettime(void)
{
    struct timeval tp;
    gettimeofday( &tp, NULL );
    
    return tp.tv_sec + 1e-6 * tp.tv_usec;
}


// *********************** CHAHID FUNCTION ROUTINES   ***********************

/*======================== Kronecker product   ===================
 __________________________________________________________________
 | This function returns the Kronecker product of A AND B           |
 | ________________________________________________________________ |*/


void kron_prod (float* kprod,  float* a, int ma, int na,  float* b, int mb, int nb)
{
    int         i, j, k, l;
    int         np = na * nb;
    
    for (i = 0; i < ma; ++ i)
        for (j = 0; j < na; ++ j)
            for (k = 0; k < mb; ++ k)
                for (l = 0; l < nb; ++ l)
                    
                    //row= (i-1)*nb+l;
                    // col=(j-1)*mb+k;
                    
                    // slice_a=(j-1)*na+i;
                    //slice_b=(k-1)*nb+l;
                    
                    
                    //c((col-1)*na*nb+row)=a(slice_a)*b(slice_b);
                   
                    kprod[(j*mb+k)*na*nb+(i*nb+l)] = a[j*na+i] * b[k*nb+l];
    
}


/*======================== unity Matrix   ===================
 __________________________________________________________________
 |              This function returns unity Matrix                  |
 | ________________________________________________________________ |*/

void Matrx_Unit(int n,int m,float *Mtx_unit){
    
    memset(Mtx_unit, 0, n*m*sizeof(float));
    
    int n2=n*m,step=0;
    
    for (int j=0; j<n2; j+=n) {
        Mtx_unit [j+step]=1;
        step++;
    }
}

/*======================== SUM matrix   ===================
 __________________________________________________________________
 |  This function returns the sum of A and B matrix  Matrix        |
 |________________________________________________________________ |*/

float SC_Sum(float h, int n,float *D1,float *D2,float *V,float *SC) {
    
    int step=0;
    float max_img0=0.0;
    
//     cout<<endl<< "-h*h="<< -h*h<<endl;
    
    for (int j=0; j<n*n; j++) {
        
        SC[j]=(-h*h)*(D1[j]+D2[j]);
        
//D         cout<<endl<< "| L = "<< D1[j]<<" + "<<D2[j];

//D         cout<<endl<< "|"<< j<<"  "<<j%(n+1)<<" == "<<SC[j];
        
        if (j%(n+1)==0){
            
            SC[j]=SC[j]-V[step];
            
            max_img0=(V[step]>max_img0?V[step]:max_img0);

//D            cout<<"- "<<V[step]<< "|";
            step++;
            
          }
        
        
    }
    return max_img0;
    
}
/*======================== Displays   ===================
 __________________________________________________________________
 | This function Shows the buffer in  Matrix representation         |
 | ________________________________________________________________ |*/

 void Disp_matrx(int n,int m,float *Mtx_A) {
   // cout<<endl<<" _______  Displaying Buffer :"<< n*m<<" Values __________"<<endl<<endl;
     
   //  for (int i=0; i<n*m; i++){
    //     printf(" | %f | ",Mtx_A [i] );}
     
cout<<endl<<endl<<" __________  Displaying Matrix :"<< n<<"X"<<m<<" _____________"<<endl<<endl;
     
    for (int i=0; i<n; i++) {
         for (int j=0; j<m; j++) {
        
                printf(" | %f | ",Mtx_A [i+j*n] );
             }
            
      printf(" \n");}
    
 }



/*======================== Displays   ===================
 __________________________________________________________________
 | This function Shows the buffer in  Matrix representation         |
 | ________________________________________________________________ |*/

void Disp_EigenValues(int n,float *Vector_A) {

    printf(" \n\n_______  Displaying Eigen Values  Vector  __________\n\n");
    
    for (int j=0; j<n; j++) {
        
        printf(" | %f | ",Vector_A[j] );
        
        
    }
    
    
    
}



/*======================= Display  Image information    =======================
 
 ___________________________________________________________________________
 |   This function displays the image information stored in  Objet_SCSA     |
 | _________________________________________________________________________|*/

bool display_Objet_SCSA( Class_SCSA *Objet_SCSA )
{
	if(msg_disp_section){

    cout<< " ============== Image Informations =============="<<endl;
    cout<< "|  Dimmension : " <<   Objet_SCSA->x_dim<<  " X "<<   Objet_SCSA->y_dim<< " Pixels."<<endl;
    //Objet_SCSA->L = 0.015915494309190;
    cout<< "|  Stored in file: "<<    Objet_SCSA->dataFileName <<" ."<<endl;
    cout<< "|*************** SCSA Parameters *****************"<<endl;
    cout<< "| h== "<<Objet_SCSA->h << "    d="<< Objet_SCSA->d<< "    gm="<<Objet_SCSA->gm << "    fe="<<Objet_SCSA->fe << "."<<endl;
    cout<< "| jobvl= "<< Objet_SCSA->jobvl << "    jobvr="<< Objet_SCSA->jobvr<< "    verbose="<< Objet_SCSA->verbose<< "."<<endl;
    cout<< " ================================================="<<endl;
    return 1;
	}
}


/*============================= SIMPEOMS'S RULE  ============================/*
 ___________________________________________________________________________
 | This function returns the numerical integration of a function f^2 using   |
 | Simpson method ot compute the  associated  L^2 -normalized eigenfunctions.|
 | _________________________________________________________________________ |*/


float simp3(float* f, int n, float dx){
    //M2      %  This function returns the numerical integration of a function f
    //M2      %  using the Simpson method
    //M2
    //M2      [n,~]=size(f);
    //M2      I=1/3*(f(1,:)+f(2,:))*dx;
    //M2
    //M2      for i=3:n
    //M2          if(mod(i,2)==0)
    //M2              I=I+(1/3*f(i,:)+1/3*f(i-1,:))*dx;
    //M2          else
    //M2              I=I+(1/3*f(i,:)+f(i-1,:))*dx;
    //M2          end
    //M2      end
    //M2      y=I;
    
    float I;
    I = (f[0]*f[0]+f[1]*f[1])*dx/3.0;
    
    for(int i = 2; i < n; i++){
        
        if (i % 2==0){
            I = I+(((1.0/3.0*f[i]*f[i])+f[i-1]*f[i-1])*dx);
            
        }
        
        else{
             I = I+(f[i]*f[i]+f[i-1]*f[i-1])*(dx/3.0);
                
                }

        
    }
    return I;
}


/*====================  EIGENVALUES DECOMPOSITION =============================
 __________________________________________________________________________________
 | This function computes all eigenvalues and, optionally, eigenvectors of :        |
 |  -> "a" real symmetric matrix of dmnsn "lda" with Lower triangle  is stored.     |
 |  -> If INFO = 0, "W" contains eigenvalues in ascending order.                    |
 |  -> If JOBZ = 'V', then if INFO = 0,A contains the orthonormal eigenvectors of A |
 |  -> if INFO = 0, WORK(1) returns the optimal LWORK                               |
 |  -> If JOBZ = 'V' and N > 1, LWORK must be at least:  1 + 6*N + 2*N**2.          |
 |  -> If JOBZ  = 'V' and N > 1, LIWORK must be at least 3 + 5*N.                   |
 |  -> INFO is INTEGER                                                              |
 |     = 0:  successful exit                                                        |
 |     < 0:  if INFO = -i, the i-th argument had an illegal value                   |
 |     > 0:  if INFO = i and JOBZ = 'N', then the algorithm failed                  |
 |              to converge; i off-diagonal elements of an intermediate tridiagonal |
 |              form did not converge to zero;                                      |
 |             if INFO = i and JOBZ = 'V', then the algorithm failed to compute an  |
 |               eigenvalue while working on the submatrix lying in rows and columns|
 |               INFO/(N+1) through  mod(INFO,N+1).                                 |
 | ________________________________________________________________________________ |*/

void syevd( const char jobz, const char uplo,
           const int n,
           float* a, const int lda, float* w,
           float* work, const int lwork, int* iwork, const int liwork, int info ){
    
    
	if(msg_disp_section){cout<<endl<<endl<<" --> Eigen Analysis of the Matrix SC_hhD . "<<endl;}
    
    ssyevd( &jobz, &uplo,
           &n,
           a, &lda, w,
           work, &lwork, iwork, &liwork, &info );
    
}





/*======================== Build the SC_hhD  matrix   ===================
 _________________________________________________________________
|   This function returns the SC_hhD  Matrix                      |
|________________________________________________________________ |*/

template<class Type> bool build_SC_matrix(float hp, int N,float *D, float *V,Type * &SC_hhD,Type *max_img){

	if(msg_disp_section){cout<<endl<<" --> The SCSA Data preparation has started. "<<endl;}

//M2
//M2 I = eye(n);
//M2 L = sparse(n*n,n*n);         % Reduce the memory starage
//M2 L = kron(I,D1) + kron(D1,I); % The 2D Laplacian operator
//M2 V = V(:) ;
//M2 SC = -h*h*L-diag(V); % The 2D Schr\"odinger operator




long int I, J, ia, ja, ib, jb,indx;
int i, k;
int step=0,N2=N*N, N4=N2*N2;
Type max_img0;
Type val;
max_img0=(Type) 0;

  int  nnz0=2*N*N*N-(N*N);
    SC_hhD = new Type[N4];
    
    


if(!SC_hhD  ){
        cout << "out of memory, could not allocate "<< N2*N2<<" of memory!" << endl;
        return false;
        }  
    else{ if(msg_disp){cout <<endl<< " --> Memory allocation ~ "<<4*N2*N2/1000 <<" Ko of memory OK!" << endl;}}

        for (I = 0; I < N2; I ++)
        for (J = 0; J < N2; J ++) { 
                    ia=I%N ;
                    ja= J%N;
                    ib=I/N ;
                    jb=J/N ;
                    
                    val=(Type) 0.0;

                    if (I==J){
                        val -= V[I];
                        if (V[I]> (float)max_img0){max_img0=(Type)V[I];}
                    } 
                    if (ia==ja ){val -= (Type) (hp*hp)*D[ib*N+jb]; }

                    if (ib==jb ){val -= (Type) (hp*hp)*D[ia*N+ja]; }

                    if (val!=(Type) 0){
                              SC_hhD[I+N2*J] = val;
                     }              
        } 

if(msg_disp){cout<<" -->  NON zeros element = "<<nnz0<< " out of "<< N2*N2<<"  => "<<100*(N*N-nnz0)/(N2*N2) <<" % of  Sparsity."<<endl<<endl;}
 
*max_img=max_img0;

 
    if(disp_matrices){
    	cout<<endl<<endl<<" --> Image  Loaded OK! "<<endl;
    	Disp_matrx( N,N, V);
        cout<<endl<<endl<<" --> D square Matrix OK! "<<endl;
        Disp_matrx( N2,N2, D);
        cout<<endl<<endl<<" --> SC_hhD square Matrix OK! "<<endl;
        Disp_matrx( N2,N2, SC_hhD);
        cout<<endl<<"----> h= "<<hp<<endl;}


return true;

}



/*============================= Get  time and date     ============================/*
 ____________________________________________________________________________________
|    This function return te excution time and date to be marked in saved files       |
|_____________________________________________________________________________________|*/

const string currentDateTime() {
    time_t     now = time(0);
    struct tm  tstruct;
    char buf[32];
    tstruct = *localtime(&now);
    // Visit http://en.cppreference.com/w/cpp/chrono/c/strftime
    // for more information about date/time format
    strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);
    return buf;
}


/*============================= Save  SCSA Performance   ============================/*
 ____________________________________________________________________________________
|    This function save the different results and input parameters in text file       |
|_____________________________________________________________________________________|*/

bool save_performance(Class_SCSA* Objet_SCSA, int N, float PSNR0, float* scsa_parmter,int sz_loop){
 
	if(msg_disp){cout<<endl<<" --> Saving the SCSA result performance.  "<<endl<<endl;}
   
FILE *fp;
int err;
char name[100];
sprintf (name, "scsa_result_%d_%0.f_%0.1f_%s_%s.txt", Objet_SCSA->x_dim,Objet_SCSA->h,PSNR0,Objet_SCSA->dataFileName,currentDateTime().c_str());

fp = fopen (name, "w");
err=fprintf(fp, " ==> SCSA_2D2D \nNoisy image : %s \nOriginal image : %s \nDone on: %s\n\n ",Objet_SCSA->dataFileName.c_str(), Objet_SCSA->originImage.c_str(), currentDateTime().c_str());

err=fprintf(fp, " N\t\t  fe\t gm\t  PSNR0\t\n ");
err=fprintf(fp, "%d\t\t %.1lf\t %d\t  %.2lf\n\n", N, Objet_SCSA->fe, Objet_SCSA->gm, PSNR0);
err=fprintf(fp, "   h\t time(s)  PSNR   Total Memory(Ko)\tNh \t \n ");

 for (int i=0; i<sz_loop; i++){          
            if (i%5==4){err=fprintf(fp, "\t\t%.0f\n ", *(scsa_parmter+i));}
            else if (i%5==3){err=fprintf(fp, "\t\t%.0f ", *(scsa_parmter+i));}
            else{
            err=fprintf(fp, "%.3f\t ", *(scsa_parmter+i));
           }
          } 

fclose(fp);

return true;
}




/*=========================== SCSA_2D2D_reconstruction   =============================
 ___________________________________________________________________________
| This function returns 2D SCSA PROCESS with paramters stored in  Objet_SCSA|
| _________________________________________________________________________ |*/

template<class Type> bool SCSA_2D2D_reconstruction(float h, int N2, int* p_Nh_size, float gm, float fe,float* matBuffer, float* Image_ref,float* SC_hhD, Type max_img, float* &V1 , double* time_Psinnor,double * time_eig, float *p_MSE0, float *p_MSE, float *p_PSNR0, float *p_PSNR){
     //*****************************************************************************************
     //M2
     //M2 % = = = = = =   The SCSA Method
     //M2 V = img;
     //M2 n = size(V,1); fe = 1; feh = 2*pi/n;
     //M2 h = 0.2550; gm = 1;
     //M2 UC = (1/(4*pi))*gamma(gm+1)/gamma(gm+2);
     //M2 h2L = h^2/UC;
     //M2
     
//      fe=1,feh=2.0*PI / N;
//      h=0.2550, gm = 1;
     //M2 D1 = Delta_1D(n,fe,feh);
     // already computed and stored in Var "D"
     //M2 I = eye(n);
     //M2 L = sparse(n*n,n*n);         % Reduce the memory starage
     //M2 L = kron(I,D1) + kron(D1,I); % The 2D Laplacian operator
     //M2
     //M2 V = V(:) ;
     //M2 SC = -h*h*L-diag(V); % The 2D Schr\"odinger operator
     //M2
 
  //M2  [psi,lambda] = eig(SC); % Eigenvalues & eigenfunction of Schr\"odinger operator

float UC = (1.0/(4.0*PI))*tgamma(gm+1.0)/tgamma(gm+2.0), h2L = (h*h)/UC;
int  i, j, N4=N2*N2, lwork = 1 + 6*N2 + 2*N4, liwork = 3 + 5*N2, iwork[liwork], info, Nh_size = 0;
float  lamda[N2] ,work[lwork];
float MSE0=0.0, MSE=0.0,PSNR0=0.0, PSNR=0.0 ;
double comp_time0, time_eig0;


if(msg_disp_section){cout<<endl<<endl<<"************************  Image Reconstruction  *************************"<< endl;}


             //M2 tic
             comp_time0 = gettime();

            // computes all eigenvalues and, optionally, eigenvectors of a
            // real symmetric matrix SC  when its Lower triangle  is stored.

             if(disp_matrices){
                          cout<<endl<<endl<<" --> SC_hhD square Matrix OK! "<<endl;
                          Disp_matrx( N2,N2, SC_hhD);}

            syevd( 'V', 'U',
                  N2,
                  SC_hhD, N2, lamda,
                  work, lwork, iwork, liwork, info );

V1 = new float[N2];
memset(V1, 0, N2*sizeof(float));
float INDX[N2], KAPPA[N2], psin[N2*Nh_size], PSINNOR[N2*Nh_size], ERR[N2],reconst_img;
memset(psin , 0, N2*Nh_size*sizeof(float));
memset(PSINNOR, 0, N2*Nh_size*sizeof(float));

            if (info == 0)
            	if(msg_disp){cout<<endl<<" The Eigenvalues and their corresponsing  orthonormal eigenvectors are computed   successfully of SC_hhD "<<endl<<endl;}

            //M2  time_eig = toc;
            time_eig0 = gettime() - comp_time0;

            if(msg_disp){cout<<endl<<" --> All Eigen values  vector OK!"<<endl<<endl;}
            //M2
            //M2  tic

            if(msg_disp_section){cout<<endl<<endl<<"*********************  EigenFunctions Normalization  ********************"<< endl;}

            //M2  % Negative eigenvalues
            //M2  temp = diag(lambda);
            //M2  ind = find(temp<0);
            //M2  kappa = diag((-temp(ind)).^(gm));
            //M2  Nh = length(kappa);
            //M2
            //M2  % The associated $L^2$-normalized eigenfunctions.
            //M2  psin = psi(:,ind(:,1));

            Nh_size = 0;

            for (int j=0;j<N2;j++)
            {

                if (lamda[j] < 0)
                {

                    KAPPA[Nh_size]=ipow(-lamda[j], gm);
                    INDX[Nh_size]=j;
                    Nh_size++;
                }

            }

//float  psin[N2*Nh_size];
//float  PSINNOR[N2*Nh_size];
//memset(psin , 0, N2*Nh_size*sizeof(float));
//memset(PSINNOR, 0, N2*Nh_size*sizeof(float));

            for (int i=0;i<N2;i++){
                for (int j=0;j<Nh_size;j++){
                    int next_psin=i + INDX[j]*N2;
                    psin[i + j*N2]=SC_hhD[next_psin];   // Compute its square
                }

            }


            //M2  psinnor = sparse(n*n,Nh);
            //M2  parfor j = 1:Nh
            //M2  %     Nh-j
            //M2      I = sqrt((simp(psin(:,j).^2,fe))^(-1));
            //M2      psinnor(:,j) = (psin(:,j).*I).^2;
            //M2  end

comp_time0 = gettime();

            for(int j = 0; j < Nh_size; j++){
                float I = 1.0/sqrt(simp3(psin+j*N2, N2, fe));
                  for(int i = 0; i < N2; i++){
                    int next_psin=i + INDX[j]*N2;
                    PSINNOR[i + j*N2]= ipow((SC_hhD[next_psin]*I),2);
                    //D            cout<<endl<<endl<<"PSINNOR["<<i + j*N2<<"]="<< PSINNOR[i + j*N2]<< "   # I="<< I<< " Psin : "<< psin[i + j*N2]<<endl;
                }
            }

            if(disp_matrices){
                cout<<endl<<" --> All  Eigen  values  vector  OK!";
                Disp_EigenValues( Nh_size, lamda);
                cout<<endl<<" --> Negative Eigen  values  vector  OK!";
                Disp_EigenValues( Nh_size, KAPPA);
                cout<<endl<<endl<<" --> Index of Negative Eigen  values  vector ";
                Disp_EigenValues( Nh_size, INDX);
                cout<<endl<<"----> Number eigen = "<<Nh_size<<endl;


                cout<<endl<<endl<<" --> Psin square Matrix : Eigen functions Matrix OK! "<<endl;
                Disp_matrx( N2,Nh_size, psin);

                cout<<endl<<endl<<" --> PSINNOR Matrix: Normalized Eigen functions Matrix OK!"<<endl;
                Disp_matrx( N2,Nh_size, PSINNOR);}

double time_Psinnor0 = gettime() - comp_time0;

            //M2
            //M2  tic
//             cout<<endl<<endl<<"************************  Image Reconstruction  *************************"<< endl;
            //M2  V1 = (h2L*sum(psinnor*kappa,2)).^(1/(1+gm));
//             float V1[N2],ERR[N2],reconst_img,MSE0=0.0,PSNR0=0.0,MSE=0.0,PSNR=0.0;
            //  memset(V1 , 0, N2*sizeof(float));

            for(int i = 0; i < N2; i++){
                reconst_img=0.0;
                for(int j = 0; j < Nh_size; j++){
                    //D               cout<<endl<<endl<<"PSINNOR["<<i + j*N2<<"]="<< PSINNOR[i + j*N2]<< "  * Kapa["<<j<<"] =:"<<KAPPA[j]<<endl;
                    reconst_img+=PSINNOR[i + j*N2]*KAPPA[j];
                }
                //D            cout<<endl<<endl<<"1/(1+gm)="<<1/(1+gm)<<endl;
                //D           cout<<endl<<endl<<"reconst_img="<<reconst_img<<endl;
                //D           cout<<endl<<endl<<"max_img="<<max_img<<endl;
//                 V1[i]=powf(reconst_img,(1/(1+gm)));
                V1[i]=powf(h2L*reconst_img,(1/(1+gm)));
                ERR[i]=((abs(matBuffer[i]-V1[i]))/max_img)*100.0;
                MSE0 += (matBuffer[i]-Image_ref[i])*(matBuffer[i]-Image_ref[i]);
                MSE += (Image_ref[i]-V1[i])*(Image_ref[i]-V1[i]);
            }

        // Evaluation PSNR comparing with the  Noisy image
        MSE0=MSE0/N2;
        PSNR0 = 10*log10(1.0/MSE0);

        // Evaliation PSNR comparing with the  Denoised image
        MSE=MSE/N2;
        PSNR = 10*log10(1.0/MSE);

memset(SC_hhD, 0, N4*sizeof(float));

*time_eig=time_eig0;
*time_Psinnor= time_Psinnor0;
*p_PSNR0=PSNR0;
*p_PSNR=PSNR;
*p_MSE0=MSE0;
*p_MSE=MSE;
*p_Nh_size=Nh_size;

if(msg_disp_section){
cout<<endl<<" --> Image Reconstruction OK!"<<endl;

cout<<endl<<endl<<"************************   2D2D SCSA Diagnosis  ***************************";

cout<<endl<<" The computations are done with the following results :  "<< endl;
cout<< "==> Image Denoising performances  for h= "<< h <<":"<<endl;
cout<<"The Noisy Image:   MSE = "<<MSE0<<  "  PSNR = "<<PSNR0<<endl;
cout<<"Denoised Image :   MSE = "<<MSE<<  "  PSNR = "<<PSNR<<endl<< endl;}

            //M2  time_sum = toc;
            //M2
            //M2  %Reshape to 2D...
            //M2  tic
            //M2  V2 = zeros(n,n);
            //M2  parfor i = 1:n
            //M2      for j = 1:n
            //M2          V2(j,i) = V1((i-1)*n+j,1);
            //M2      end
            //M2  end
            //M2  time_reshape = toc;
            //M2  img
            //M2  Budder_img=V'
            //M2  SCSA_img=V2
            //M2  ERR =(abs(img-V2))./max(max(img)).*100
            //M2  MSE = mean2((img - V2).^2);
            //M2  PSNR = 10*log10(1/MSE);
            //M2
            //M2  time_all = toc;
            //M2  psnr_msg=strcat(' MSE =',num2str(MSE),' PSNR =   ',num2str(PSNR))
 return true;
}

/*   Comment
  //EX  : displays during execution
 
  //D1  :
 
 
 */

