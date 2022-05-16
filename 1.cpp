#include<iostream>
#include<sys/time.h>

#include<omp.h>

using namespace std;

int n;
float** A;
struct timeval val;
struct timeval newval;

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
	#pragma omp parallel
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


int main()
{
    //cin>>n;  //矩阵规模
    int step=10;
    for(n=0;n<=2000;n+=step)
    {
	for(int i=0;i<n;i++)
		A = new float*[n];
    for(int i=0;i<n;i++)
		A[i] = new float[n];
    double time_0=0.0,time_1=0.0;
    //串行算法
	m_reset();
	int ret=gettimeofday(&val,NULL);  
    normal();
	ret=gettimeofday(&newval,NULL);
	time_0+=(newval.tv_sec - val.tv_sec) + (double)(newval.tv_usec - val.tv_usec) / 1000000.0; 
	//openMP
	m_reset();
	ret=gettimeofday(&val,NULL);  
    p_openMP();
	ret=gettimeofday(&newval,NULL);
	time_1+=(newval.tv_sec - val.tv_sec) + (double)(newval.tv_usec - val.tv_usec) / 1000000.0; 
    cout<<"        "<<n<<"        "<<"& "<<time_0<<"   "<<"& "<<time_1<<" "<<R"(\\ \hline)"<<endl;
    for(int i=0;i<n;i++)
        delete A[i];
    delete A;
    if(n==100){step=100;}
    if(n==1000){step=1000;}

    }
	//print_result();
	return 0;
} 

