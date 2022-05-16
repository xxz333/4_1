#include<iostream>
#include<sys/time.h>

#include<omp.h>

#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>

using namespace std;

int n;
float** A;
struct timeval val;
struct timeval newval;
//线程数
const int num=7;

void m_reset()
{
	for(int i=0;i<n;i++){
		for(int j=0;j<i;j++)
			A[i][j]=0;
		A[i][i]=1.0;
		for(int j=i+1;j<n;j++)
			A[i][j]=rand()%100;
	}
	for(int k=0;k<n;k++)
		for(int i=k+1;i<n;i++)
			for(int j=0;j<n;j++)
				A[i][j]+=A[k][j];
}

void print_result()
{
	for(int i=0;i<n;i++)
		for(int j=0;j<n;j++)
			cout<<A[i][j]<<" ";
		cout<<endl;
}

//串行算法
void normal()
{
	for(int k=0;k<n;++k)
	{
		for(int j=k+1;j<n;j++)   
                	A[k][j]=A[k][j]/A[k][k];
        	A[k][k]=1.0;
	    for(int i = k + 1; i < n; ++i)
        {
			float tmp = A[i][k];
			for(int j = k + 1; j < n; ++j)
            {
				A[i][j] = A[i][j] - tmp * A[k][j];
			}
			A[i][k] = 0.0;
        }
	}
}
//openMP
void p_openMP()
{
	//创建线程
	#pragma omp parallel num_threads(num)
	for(int k=0;k<n;++k)
	{
		// 串行部分
		#pragma omp single//只交给一个线程来执行
		{
			float tmp = A[k][k];
			for(int j=k+1;j<n;j++)   
                	A[k][j]=A[k][j]/tmp;
        	A[k][k]=1.0;
		}
		//并行部分，使用行划分
		#pragma omp for//之后的for循环将被并行化由多个线程划分执行
	    for(int i = k + 1; i < n; ++i)
        {
			float tmp = A[i][k];
			for(int j = k + 1; j < n; ++j)
            {
				A[i][j] = A[i][j] - tmp * A[k][j];
			}
			A[i][k] = 0.0;
        }
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

//openMP+sse+不对齐
void p_openMP_sse_n()
{
	//创建线程
	#pragma omp parallel num_threads(num)
	for(int k=0;k<n;++k)
	{
		// 串行部分
		#pragma omp single//只交给一个线程来执行
		{
			__m128 vt=_mm_set1_ps(A[k][k]);
            int j;
            for(j=k+1;j+4<n;j+=4)
            {
                __m128 va=_mm_loadu_ps(A[k]+j);
				va=_mm_div_ps(va,vt);
				_mm_storeu_ps(A[k]+j,va);
            }
            //处理剩下的部分
            for(;j<n;j++)
                A[k][j]=A[k][j]/A[k][k];
            A[k][k]=1.0;
		}
		//并行部分，使用行划分
		#pragma omp for//之后的for循环将被并行化由多个线程划分执行
	    for(int i = k + 1; i < n; ++i)
        {
			__m128 vaik=_mm_set1_ps(A[i][k]);
            int j;
            for(j=k+1;j+4<n;j+=4)
            {
                __m128 vakj=_mm_loadu_ps(A[k]+j);
                __m128 vaij=_mm_loadu_ps(A[i]+j);
                __m128 vx=_mm_mul_ps(vaik,vakj);
                vaij=_mm_sub_ps(vaij,vx);
                _mm_storeu_ps(&A[i][j],vaij);
            }
			for(; j < n; ++j)
            {
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0.0;
        }
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

//openMP+sse+内存对齐
void p_openMP_sse()
{
	//创建线程
	#pragma omp parallel num_threads(num)
	for(int k=0;k<n;++k)
	{
		// 串行部分
		#pragma omp single//只交给一个线程来执行
		{
			__m128 vt=_mm_set1_ps(A[k][k]);
            int j=k+1;
            while((j%4!=0)&&(j<n))
            {
                A[k][j]=A[k][j]/A[k][k];
                j++;
            }
            for(;j+4<n;j+=4)
            {
                __m128 va=_mm_loadu_ps(A[k]+j);
				va=_mm_div_ps(va,vt);
				_mm_storeu_ps(A[k]+j,va);
            }
            //处理剩下的部分
            for(;j<n;j++)
                A[k][j]=A[k][j]/A[k][k];
            A[k][k]=1.0;
		}
		//并行部分，使用行划分
		#pragma omp for//之后的for循环将被并行化由多个线程划分执行
	    for(int i = k + 1; i < n; ++i)
        {
			__m128 vaik=_mm_set1_ps(A[i][k]);
            int j=k+1;
            while((j%4!=0)&&(j<n))
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
                j++;
            }
            for(;j+4<n;j+=4)
            {
                __m128 vakj=_mm_loadu_ps(A[k]+j);
                __m128 vaij=_mm_loadu_ps(A[i]+j);
                __m128 vx=_mm_mul_ps(vaik,vakj);
                vaij=_mm_sub_ps(vaij,vx);
                _mm_storeu_ps(&A[i][j],vaij);
            }
			for(; j < n; ++j)
            {
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0.0;
        }
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

//openMP+avx+不对齐
void p_openMP_avx_n()
{
	//创建线程
	#pragma omp parallel num_threads(num)
	for(int k=0;k<n;++k)
	{
		// 串行部分
		#pragma omp single//只交给一个线程来执行
		{
			__m256 vt=_mm256_set1_ps(A[k][k]);
            int j;
            for(j=k+1;j+8<n;j+=8)
            {
                __m256 va=_mm256_loadu_ps(A[k]+j);
				va=_mm256_div_ps(va,vt);
				_mm256_storeu_ps(A[k]+j,va);
            }
            //处理剩下的部分
            for(;j<n;j++)
                A[k][j]=A[k][j]/A[k][k];
            A[k][k]=1.0;
		}
		//并行部分，使用行划分
		#pragma omp for//之后的for循环将被并行化由多个线程划分执行
	    for(int i = k + 1; i < n; ++i)
        {
			__m256 vaik=_mm256_set1_ps(A[i][k]);
            int j;
            for(j=k+1;j+8<n;j+=8)
            {
                __m256 vakj=_mm256_loadu_ps(A[k]+j);
                __m256 vaij=_mm256_loadu_ps(A[i]+j);
                __m256 vx=_mm256_mul_ps(vaik,vakj);
                vaij=_mm256_sub_ps(vaij,vx);
                _mm256_storeu_ps(&A[i][j],vaij);
            }
			for(; j < n; ++j)
            {
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0.0;
        }
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

//openMP+avx+对齐
void p_openMP_avx()
{
    //创建线程
	#pragma omp parallel num_threads(num)
	for(int k=0;k<n;++k)
	{
		// 串行部分
		#pragma omp single//只交给一个线程来执行
		{
			__m256 vt=_mm256_set1_ps(A[k][k]);
            int j=k+1;
            while((j%8!=0)&&(j<n))
            {
                A[k][j]=A[k][j]/A[k][k];
                j++;
            }
            for(;j+8<n;j+=8)
            {
                __m256 va=_mm256_loadu_ps(A[k]+j);
				va=_mm256_div_ps(va,vt);
				_mm256_storeu_ps(A[k]+j,va);
            }
            //处理剩下的部分
            for(;j<n;j++)
                A[k][j]=A[k][j]/A[k][k];
            A[k][k]=1.0;
		}
		//并行部分，使用行划分
		#pragma omp for//之后的for循环将被并行化由多个线程划分执行
	    for(int i = k + 1; i < n; ++i)
        {
			__m256 vaik=_mm256_set1_ps(A[i][k]);
            int j=k+1;
            while((j%8!=0)&&(j<n))
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
                j++;
            }
            for(;j+8<n;j+=8)
            {
                __m256 vakj=_mm256_loadu_ps(A[k]+j);
                __m256 vaij=_mm256_loadu_ps(A[i]+j);
                __m256 vx=_mm256_mul_ps(vaik,vakj);
                vaij=_mm256_sub_ps(vaij,vx);
                _mm256_storeu_ps(&A[i][j],vaij);
            }
			for(; j < n; ++j)
            {
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0.0;
        }
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

//openMP+avx512+不对齐
void p_openMP_avx512_n()
{
	//创建线程
	#pragma omp parallel num_threads(num)
	for(int k=0;k<n;++k)
	{
		// 串行部分
		#pragma omp single//只交给一个线程来执行
		{
			__m512 vt=_mm512_set1_ps(A[k][k]);
            int j;
            for(j=k+1;j+16<n;j+=16)
            {
                __m512 va=_mm512_loadu_ps(A[k]+j);
				va=_mm512_div_ps(va,vt);
				_mm512_storeu_ps(A[k]+j,va);
            }
            //处理剩下的部分
            for(;j<n;j++)
                A[k][j]=A[k][j]/A[k][k];
            A[k][k]=1.0;
		}
		//并行部分，使用行划分
		#pragma omp for//之后的for循环将被并行化由多个线程划分执行
	    for(int i = k + 1; i < n; ++i)
        {
			__m512 vaik=_mm512_set1_ps(A[i][k]);
            int j;
            for(j=k+1;j+16<n;j+=16)
            {
                __m512 vakj=_mm512_loadu_ps(A[k]+j);
                __m512 vaij=_mm512_loadu_ps(A[i]+j);
                __m512 vx=_mm512_mul_ps(vaik,vakj);
                vaij=_mm512_sub_ps(vaij,vx);
                _mm512_storeu_ps(&A[i][j],vaij);
            }
			for(; j < n; ++j)
            {
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0.0;
        }
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

//openMP+avx512+对齐
void p_openMP_avx512()
{
    //创建线程
	#pragma omp parallel num_threads(num)
	for(int k=0;k<n;++k)
	{
		// 串行部分
		#pragma omp single//只交给一个线程来执行
		{
			__m512 vt=_mm512_set1_ps(A[k][k]);
            int j=k+1;
            while((j%16!=0)&&(j<n))
            {
                A[k][j]=A[k][j]/A[k][k];
                j++;
            }
            for(;j+16<n;j+=16)
            {
                __m512 va=_mm512_loadu_ps(A[k]+j);
				va=_mm512_div_ps(va,vt);
				_mm512_storeu_ps(A[k]+j,va);
            }
            //处理剩下的部分
            for(;j<n;j++)
                A[k][j]=A[k][j]/A[k][k];
            A[k][k]=1.0;
		}
		//并行部分，使用行划分
		#pragma omp for//之后的for循环将被并行化由多个线程划分执行
	    for(int i = k + 1; i < n; ++i)
        {
			__m512 vaik=_mm512_set1_ps(A[i][k]);
            int j=k+1;
            while((j%16!=0)&&(j<n))
            {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
                j++;
            }
            for(;j+16<n;j+=16)
            {
                __m512 vakj=_mm512_loadu_ps(A[k]+j);
                __m512 vaij=_mm512_loadu_ps(A[i]+j);
                __m512 vx=_mm512_mul_ps(vaik,vakj);
                vaij=_mm512_sub_ps(vaij,vx);
                _mm512_storeu_ps(&A[i][j],vaij);
            }
			for(; j < n; ++j)
            {
				A[i][j] = A[i][j] - A[i][k] * A[k][j];
			}
			A[i][k] = 0.0;
        }
		// 离开for循环时，各个线程默认同步，进入下一行的处理
	}
}

int main()
{
    //cin>>n;  //矩阵规模
    int step=10;
    for(n=0;n<=10000;n+=step)
    {
	for(int i=0;i<n;i++)
		A = new float*[n];
    for(int i=0;i<n;i++)
		A[i] = new float[n];
    double time_0=0.0,time_1=0.0,time_2=0.0,time_3=0.0,time_4=0.0;
    double time_2_2=0.0,time_3_2=0.0,time_4_2=0.0;
        
	//openMP
	m_reset();
	gettimeofday(&val,NULL);  
    p_openMP();
	gettimeofday(&newval,NULL);
	time_1+=(newval.tv_sec - val.tv_sec) + (double)(newval.tv_usec - val.tv_usec) / 1000000.0; 
    
    //openMP+sse+不对齐
	m_reset();
	gettimeofday(&val,NULL);  
    p_openMP_sse_n();
	gettimeofday(&newval,NULL);
	time_2+=(newval.tv_sec - val.tv_sec) + (double)(newval.tv_usec - val.tv_usec) / 1000000.0;
        
    //openMP+sse+对齐
	m_reset();
	gettimeofday(&val,NULL);  
    p_openMP_sse();
	gettimeofday(&newval,NULL);
	time_2_2+=(newval.tv_sec - val.tv_sec) + (double)(newval.tv_usec - val.tv_usec) / 1000000.0;
        
    //openMP+avx
	m_reset();
	gettimeofday(&val,NULL);  
    p_openMP_avx_n();
	gettimeofday(&newval,NULL);
	time_3+=(newval.tv_sec - val.tv_sec) + (double)(newval.tv_usec - val.tv_usec) / 1000000.0; 
        
    //openMP+avx+对齐
    m_reset();
	gettimeofday(&val,NULL);  
    p_openMP_avx();
	gettimeofday(&newval,NULL);
	time_3_2+=(newval.tv_sec - val.tv_sec) + (double)(newval.tv_usec - val.tv_usec) / 1000000.0;
        
    //openMP+avx512
	m_reset();
	gettimeofday(&val,NULL);  
    p_openMP_avx512_n();
	gettimeofday(&newval,NULL);
	time_4+=(newval.tv_sec - val.tv_sec) + (double)(newval.tv_usec - val.tv_usec) / 1000000.0; 
        
    //openMP+avx512+对齐
	m_reset();
	gettimeofday(&val,NULL);  
    p_openMP_avx512();
	gettimeofday(&newval,NULL);
	time_4_2+=(newval.tv_sec - val.tv_sec) + (double)(newval.tv_usec - val.tv_usec) / 1000000.0; 
    
    cout<<"        "<<n<<"        "<<"& "<<time_1<<"   "<<"& "<<time_2<<"   "<<"& "<<time_3
        <<"   "<<"& "<<time_4<<"   "<<"& "<<time_2_2<<"   "<<"& "<<time_3_2<<"   "<<"& "<<time_4_2<<"   "<<R"(\\ \hline)"<<endl;
    for(int i=0;i<n;i++)
        delete A[i];
    delete A;
    if(n==100){step=100;}
    if(n==1000){step=1000;}

    }
	//print_result();
	return 0;
} 

