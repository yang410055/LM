

#include <iostream>
using namespace std;

#include <vector>
#include <math.h>

#include <opencv2/core/core.hpp>
//using namespace cv;

const double DERIV_STEP = 1e-5;
const int MAX_ITER = 1000;



void LM( double(*Func)( const cv::Mat &input, const cv::Mat pms ),  
	                               const cv::Mat &inputs, const cv::Mat &outputs, cv::Mat &pms);

double Deriv( double(*Func)( const cv::Mat &input, const cv::Mat pms),
                                    const cv::Mat &input, const cv::Mat pms, int n  );

double Func(const cv::Mat &input, const cv::Mat pms);


int main()
{
	
	//参数数量
	int num_params = 4;

	//数据数量
	int num_data = 100;

	cv::Mat input( num_data, 1, CV_64F );
	cv::Mat output( num_data, 1, CV_64F );

	double A = 5;
	double B = 1;
	double C = 10;
	double D = 2;

	for (int i = 0; i < num_data; i++)
	{
		double x = -10.0 + 20.0*rand()/(RAND_MAX+1);
		double y = A * sin(B*x) + C * cos(D*x);

		input.at<double>(i, 0) = x;
		output.at<double>(i, 0) = y;

	}

	cv::Mat params(num_params,1,CV_64F );
	//赋初值
	params.at<double>(0, 0) = 1;
	params.at<double>(1, 0) = 1;
	params.at<double>(2, 0) = 8;
	params.at<double>(3, 0) = 1;

	LM(Func, input, output, params);

	cout << params << endl;
	
	int ab = 1;

	return 1;

}


double Func(const cv::Mat &input, const cv::Mat pms)
{
	double A = pms.at<double>(0, 0);
	double B = pms.at<double>(1, 0);
	double C = pms.at<double>(2, 0);
	double D = pms.at<double>(3, 0);

	double x = input.at<double>(0, 0);

	double result = A * sin(B*x) + C * cos(D*x);
	
	return result;

}

double Deriv(double(*Func)(const cv::Mat &input, const cv::Mat pms),
	const cv::Mat &input, const cv::Mat pms, int n)
{
	cv::Mat pms1 = pms.clone();
	cv::Mat pms2 = pms.clone();


	pms1.at<double>(n, 0) -= DERIV_STEP;
	pms2.at<double>(n, 0) += DERIV_STEP;

	double p1 = Func(input, pms1);
	double p2 = Func(input, pms2);

	double result = (p2 - p1) / (2*DERIV_STEP);

	return result;
}


void LM(double(*Func)(const cv::Mat &input, const cv::Mat pms),
	const cv::Mat &inputs, const cv::Mat &outputs, cv::Mat &pms)
{
	int m = inputs.rows;
	int n = inputs.cols;
	int num_params = pms.rows;

	cv::Mat r(m, 1, CV_64F);  //residual
	cv::Mat r_tmp(m, 1, CV_64F);

	cv::Mat J(m, num_params, CV_64F );

	cv::Mat single_row_input(1, n, CV_64F);

	cv::Mat params_tmp = pms.clone();

	double last_mse = 0;
	float u = 1, v = 2;

	cv::Mat I = cv::Mat::ones(num_params, num_params, CV_64F);

	for (int i = 0; i < MAX_ITER; i++)
	{
		double mse = 0;
		double mse_temp = 0;

		for (int j = 0; j < m; j++)  //对应每一组数据，inputs的每一行是一组新数据，对应一组outputs，outputs一般一组只有一个值
		{
			//取一组输出
			for (int k = 0; k < n; k++)
			{
				single_row_input.at<double>(0, k) = inputs.at<double>(j, k);
			}

			//计算residual
			r.at<double>(j,0) = outputs.at<double>(j, 0) - Func(single_row_input, pms);
			mse = r.at<double>(j, 0)*r.at<double>(j,0);

			//构造雅克比矩阵
			for (int k = 0; k < num_params; k++)
			{
				J.at<double>(j, k) = Deriv(Func, single_row_input, pms, k);
			}

		}

		mse /= m;
		params_tmp = pms.clone();

		cv::Mat hlm = (J.t()*J + u * I).inv()*J.t()*r;   //应该为-
		params_tmp += hlm;

		for (int j = 0; j < m; j++)
		{
			r_tmp.at<double>(j, 0) = outputs.at<double>(j, 0) - Func(single_row_input, params_tmp);
			mse_temp += r_tmp.at<double>(j, 0)*r_tmp.at<double>(j,0);
		}
		mse_temp /= m;

		cv::Mat q(1,1,CV_64F);
		q = (mse - mse_temp) / (0.5*hlm.t()*(u*hlm-J.t()*r));
		double q_value = q.at<double>(0,0);

		if (q_value > 0)
		{
			double s = 1.0 / 3.0;
			v = 2;
			mse = mse_temp;
			pms = params_tmp;

			double temp = 1 - pow(2*q_value-1,3);
			if (s > temp)
			{
				u = u * s;
			}
			else
			{
				u = u * temp;
			}
		}
		else
		{
			u = u * v;
			v = 2 * v;
			pms = params_tmp;
		}


		if (fabs(mse - last_mse) < 1e-8)
		{
			break;
		}

		cout << i << " " << mse << endl;
		last_mse = mse;
	}


	




}