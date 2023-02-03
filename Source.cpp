#include <iostream>
#include <iomanip>
#include <sstream>
#include <numbers>
#include <vector>
#include <omp.h>
#include "BMPFileRW.h"
#include <tbb/tbb.h>
#include <math.h>

#define ExpCount 10

using namespace std;

void fillingMatrCoef(double**& MATR_coef, int RH, int RW, double sigma);

// Линейная фильтрация методом Гаусса с использованием omp parallel for
double LinearGaussFilter(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int imWidth, int  imHeight, int RH, int RW)
{
	double time_start = omp_get_wtime();
	
	int M = 2 * RH + 1, N = 2 * RW + 1;
	double** MATR_coef = new double* [M];
	MATR_coef[0] = new double[M * N];
	for (int i = 1; i < M; i++)
	{
		MATR_coef[i] = &MATR_coef[0][N * i];
	}

	fillingMatrCoef(MATR_coef, RH, RW, RH);

#pragma omp parallel for
	for (int Y = 0; Y < imHeight; Y++)
		for (int X = 0; X < imWidth; X++)
		{
			int LinF_Value_R = 0, LinF_Value_G = 0, LinF_Value_B = 0;
			for (int DY = -RH; DY <= RH; DY++)
			{
				int KY = Y + DY;
				if (KY < 0)
					KY = 0;
				if (KY > imHeight - 1)
					KY = imHeight - 1;
				for (int DX = -RW; DX <= RW; DX++)
				{
					int KX = X + DX;
					if (KX < 0)
						KX = 0;
					if (KX > imWidth - 1)
						KX = imWidth - 1;
					LinF_Value_R += rgb_in[KY][KX].rgbtRed * MATR_coef[DY + RH][DX + RW];
					LinF_Value_G += rgb_in[KY][KX].rgbtGreen * MATR_coef[DY + RH][DX + RW];
					LinF_Value_B += rgb_in[KY][KX].rgbtBlue * MATR_coef[DY + RH][DX + RW];
				}
			}
			vector<int> LinF_Values{ LinF_Value_R , LinF_Value_G ,LinF_Value_B };

			for (auto it : LinF_Values)
			{
				if (it < 0)
					it = 0;
				if (it > 255)
					it = 255;
			}
			rgb_out[Y][X].rgbtRed = LinF_Value_R;
			rgb_out[Y][X].rgbtGreen = LinF_Value_G;
			rgb_out[Y][X].rgbtBlue = LinF_Value_B;
		}

	return (omp_get_wtime() - time_start) * 1000;
}

// Линейная фильтрация методом Гаусса с использованием tbb::parallel_for
double LinearGaussFilterTBB(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int imWidth, int  imHeight, int RH, int RW)
{
	double time_start = omp_get_wtime();

	int M = 2 * RH + 1, N = 2 * RW + 1;
	double** MATR_coef = new double* [M];
	MATR_coef[0] = new double[M * N];
	for (int i = 1; i < M; i++)
	{
		MATR_coef[i] = &MATR_coef[0][N * i];
	}

	fillingMatrCoef(MATR_coef, RH, RW, 1.8);

	tbb::parallel_for(tbb::blocked_range2d<int>(0, imHeight, 0, imWidth), [&](tbb::blocked_range2d<int> r)
		{
			for (int Y = r.rows().begin(); Y < r.rows().end(); Y++)
			for (int X = r.cols().begin(); X < r.cols().end(); X++)
			{
				int LinF_Value_R = 0, LinF_Value_G = 0, LinF_Value_B = 0;
				for (int DY = -RH; DY <= RH; DY++)
				{
					int KY = Y + DY;
					if (KY < 0)
						KY = 0;
					if (KY > imHeight - 1)
						KY = imHeight - 1;
					for (int DX = -RW; DX <= RW; DX++)
					{
						int KX = X + DX;
						if (KX < 0)
							KX = 0;
						if (KX > imWidth - 1)
							KX = imWidth - 1;
						LinF_Value_R += rgb_in[KY][KX].rgbtRed * MATR_coef[DY + RH][DX + RW];
						LinF_Value_G += rgb_in[KY][KX].rgbtGreen * MATR_coef[DY + RH][DX + RW];
						LinF_Value_B += rgb_in[KY][KX].rgbtBlue * MATR_coef[DY + RH][DX + RW];
					}
				}
				vector<int> LinF_Values{ LinF_Value_R , LinF_Value_G ,LinF_Value_B };
				for (auto it : LinF_Values)
				{
					if (it < 0)
						it = 0;
					if (it > 255)
						it = 255;
				}
				rgb_out[Y][X].rgbtRed = LinF_Value_R;
				rgb_out[Y][X].rgbtGreen = LinF_Value_G;
				rgb_out[Y][X].rgbtBlue = LinF_Value_B;
			}
		});

	return (omp_get_wtime() - time_start) * 1000;
}

// Функция заполнения коэффициентов
void fillingMatrCoef(double**& MATR_coef, int RH, int RW, double sigma)
{
	double SUM = 0;
	for (int Y = -RH; Y <= RH; Y++)
	{
		for (int X = -RW; X <= RW; X++)
		{
			int YK = Y + RH, XK = X + RW;
			double CF = (1 / (2 * numbers::pi * pow(sigma, 2.))) * exp(-1 * (pow(X, 2) + pow(Y, 2)) / (2 * pow(sigma, 2)));
			MATR_coef[YK][XK] = CF;
			SUM += MATR_coef[YK][XK];
		}
	}
	for (int Y = -RH; Y <= RH; Y++)
	{
		for (int X = -RW; X <= RW; X++)
		{
			int YK = Y + RH, XK = X + RW;
			MATR_coef[YK][XK] = MATR_coef[YK][XK] * (1 / SUM);
		}
	}
}

// Линейная фильтрация Среднеарифметическим фильтром с использованием omp parallel for
double LinearArithmeticMeanFilter(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int imWidth, int  imHeight, int RH, int RW)
{
	double time_start = omp_get_wtime();

	int size = (RH * 2 + 1) * (RW * 2 + 1);
#pragma omp parallel for
	for (int Y = 0; Y < imHeight; Y++)
	{
		for (int X = 0; X < imWidth; X++)
		{
			int Summ_Value_R = 0, Summ_Value_G = 0, Summ_Value_B = 0;

			for (int DY = -RH; DY <= RH; DY++)
			{
				int KY = Y + DY;
				if (KY < 0)
					KY = 0;
				if (KY > imHeight - 1)
					KY = imHeight - 1;
				for (int DX = -RW; DX <= RW; DX++)
				{
					int KX = X + DX;
					if (KX < 0)
						KX = 0;
					if (KX > imWidth - 1)
						KX = imWidth - 1;
					Summ_Value_R += rgb_in[KY][KX].rgbtRed;
					Summ_Value_G += rgb_in[KY][KX].rgbtGreen;
					Summ_Value_B += rgb_in[KY][KX].rgbtBlue;
				}
			}
			rgb_out[Y][X].rgbtRed = Summ_Value_R / size;
			rgb_out[Y][X].rgbtGreen = Summ_Value_G / size;
			rgb_out[Y][X].rgbtBlue = Summ_Value_B / size;
		}
	}
	return (omp_get_wtime() - time_start) * 1000;
}

// Линейная фильтрация Среднеарифметическим фильтром с использованием tbb::parallel_for
double LinearArithmeticMeanFilterTBB(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int imWidth, int  imHeight, int RH, int RW)
{
	double time_start = omp_get_wtime();

	int size = (RH * 2 + 1) * (RW * 2 + 1);

	tbb::parallel_for(tbb::blocked_range2d<int>(0, imHeight, 0, imWidth), [&](tbb::blocked_range2d<int> r)
		{
			for (int Y = r.rows().begin(); Y < r.rows().end(); Y++)
				for (int X = r.cols().begin(); X < r.cols().end(); X++)
				{
					int Summ_Value_R = 0, Summ_Value_G = 0, Summ_Value_B = 0;

					for (int DY = -RH; DY <= RH; DY++)
					{
						int KY = Y + DY;
						if (KY < 0)
							KY = 0;
						if (KY > imHeight - 1)
							KY = imHeight - 1;
						for (int DX = -RW; DX <= RW; DX++)
						{
							int KX = X + DX;
							if (KX < 0)
								KX = 0;
							if (KX > imWidth - 1)
								KX = imWidth - 1;
							Summ_Value_R += rgb_in[KY][KX].rgbtRed;
							Summ_Value_G += rgb_in[KY][KX].rgbtGreen;
							Summ_Value_B += rgb_in[KY][KX].rgbtBlue;
						}
					}
					rgb_out[Y][X].rgbtRed = Summ_Value_R / size;
					rgb_out[Y][X].rgbtGreen = Summ_Value_G / size;
					rgb_out[Y][X].rgbtBlue = Summ_Value_B / size;
				}
		});
	return (omp_get_wtime() - time_start) * 1000;
}

// Функция рассчёта текстурных признаков с использованием omp parallel for 
double texturefeatures(RGBTRIPLE**& rgb_in, int &imWidth, int  &imHeight, int &RH, int &RW)
{
	double t_start = omp_get_wtime();
	int size = (RH * 2 + 1) * (RW * 2 + 1);

	RGBTRIPLE** rgb_out_m2 = new RGBTRIPLE * [imHeight], ** rgb_out_r = new RGBTRIPLE * [imHeight],
		** rgb_out_u = new RGBTRIPLE * [imHeight], ** rgb_out_e = new RGBTRIPLE * [imHeight];

	rgb_out_m2[0] = new RGBTRIPLE[imWidth * imHeight];
	rgb_out_u[0] = new RGBTRIPLE[imWidth * imHeight];
	rgb_out_r[0] = new RGBTRIPLE[imWidth * imHeight];
	rgb_out_e[0] = new RGBTRIPLE[imWidth * imHeight];

	for (int i = 1; i < imHeight; i++)
	{
		rgb_out_m2[i] = &rgb_out_m2[0][imWidth * i];
		rgb_out_u[i] = &rgb_out_u[0][imWidth * i];
		rgb_out_r[i] = &rgb_out_r[0][imWidth * i];
		rgb_out_e[i] = &rgb_out_e[0][imWidth * i];
	}

	int** bimage = new int* [imHeight];
	double **M2 = new double* [imHeight],
		**R = new double* [imHeight], **U = new double* [imHeight],
		**E = new double* [imHeight];

	for (int i = 0; i < imHeight; i++)
	{
		bimage[i] = new int[imWidth];
		M2[i] = new double[imWidth];
		R[i] = new double[imWidth];
		U[i] = new double[imWidth];
		E[i] = new double[imWidth];
	}

#pragma omp parallel for schedule(dynamic)
	for (int Y = 0; Y < imHeight; Y++)
		for (int X = 0; X < imWidth; X++)
			bimage[Y][X] = rgb_in[Y][X].rgbtRed * 0.299 + rgb_in[Y][X].rgbtGreen * 0.587 + rgb_in[Y][X].rgbtBlue * 0.114;

	double max_r = 0, max_m2 = 0, max_e = 0, max_u = 0, min_m2 = 0, min_u = 0, min_e = 0, min_r = 0;

#pragma omp parallel for schedule(dynamic)
		for (int Y = 0; Y < imHeight; Y++)
		{
			double hist[256];
			for (int X = 0; X < imWidth; X++)
			{
				for (int i = 0; i <= 255; i++)
					hist[i] = 0;

				for (int DY = -RH; DY <= RH; DY++)
				{
					int KY = Y + DY;
					if (KY < 0)
						KY = 0;
					if (KY > imHeight - 1)
						KY = imHeight - 1;
					for (int DX = -RW; DX <= RW; DX++)
					{
						int KX = X + DX;
						if (KX < 0)
							KX = 0;
						if (KX > imWidth - 1)
							KX = imWidth - 1;
						hist[bimage[KY][KX]] = hist[bimage[KY][KX]] + 1;
					}
				}

				double m = 0, m2 = 0, e = 0, u = 0;

				for (int i = 0; i <= 255; i++)
					hist[i] = hist[i] / size;
				for (int i = 0; i <= 255; i++)
					m += hist[i] * i;
				for (int i = 0; i <= 255; i++)
				{
					m2 += pow(i - m, 2) * hist[i];
					if (hist[i] != 0)
						e += hist[i] * log2(hist[i]);
					u += pow(hist[i], 2);
				}
				M2[Y][X] = m2;
				U[Y][X] = u;
				R[Y][X] = 1 - (1 / (1 + m2));
				E[Y][X] = -1 * e;

				if (Y == 0 && X == 0)
				{
					max_m2 = m2;
					max_u = u;
					max_r = (1 - (1 / (1 + m2)));
					max_e = (-1 * e);
					min_m2 = m2;
					min_u = u;
					min_r = (1 - (1 / (1 + m2)));
					min_e = (-1 * e);
				}
				else
				{
					if (M2[Y][X] > max_m2)
						max_m2 = M2[Y][X];
					if (R[Y][X] > max_r)
						max_r = R[Y][X];
					if (U[Y][X] > max_u)
						max_u = U[Y][X];
					if (E[Y][X] > max_e)
						max_e = E[Y][X];

					if (m2 < min_m2)
						min_m2 = m2;
					if (R[Y][X] < min_r)
						min_r = R[Y][X];
					if (u < min_u)
						min_u = u;
					if (-1 * e < min_e)
						min_e = -1 * e;
				}
			}
		}
	double T1_m2 = (max_m2 - min_m2) * 0.3 + min_m2;
	double T2_m2 = (max_m2 - min_m2) * 0.7 + min_m2;

	double T1_u = (max_u - min_u) * 0.3 + min_u;
	double T2_u = (max_u - min_u) * 0.7 + min_u;

	double T1_e = (max_e - min_e) * 0.3 + min_e;
	double T2_e = (max_e - min_e) * 0.7 + min_e;

	double T1_r = (max_r - min_r) * 0.3 + min_r;
	double T2_r = (max_r - min_r) * 0.7 + min_r;

#pragma omp parallel for schedule(dynamic)
	for (int Y = 0; Y < imHeight; Y++)
	{
		for (int X = 0; X < imWidth; X++)
		{
			if (M2[Y][X] <= T1_m2)
			{
				rgb_out_m2[Y][X].rgbtGreen = 255;
			}
			else if (T1_m2 < M2[Y][X] && M2[Y][X] < T2_m2)
			{
				rgb_out_m2[Y][X].rgbtRed = 255;
				rgb_out_m2[Y][X].rgbtGreen = 255;
			}
			else
			{
				rgb_out_m2[Y][X].rgbtRed = 255;
			}

			if (U[Y][X] <= T1_u)
			{
				rgb_out_u[Y][X].rgbtGreen = 255;
			}
			else if (T1_u < U[Y][X] && U[Y][X] < T2_u)
			{
				rgb_out_u[Y][X].rgbtRed = 255;
				rgb_out_u[Y][X].rgbtGreen = 255;
			}
			else
			{
				rgb_out_u[Y][X].rgbtRed = 255;
			}

			if (R[Y][X] <= T1_r)
			{
				rgb_out_r[Y][X].rgbtGreen = 255;
			}
			else if (T1_r < R[Y][X] && R[Y][X] < T2_r)
			{
				rgb_out_r[Y][X].rgbtRed = 255;
				rgb_out_r[Y][X].rgbtGreen = 255;
			}
			else
			{
				rgb_out_r[Y][X].rgbtRed = 255;
			}

			if (E[Y][X] <= T1_e)
			{
				rgb_out_e[Y][X].rgbtGreen = 255;
			}
			else if (T1_e < E[Y][X] && E[Y][X] < T2_e)
			{
				rgb_out_e[Y][X].rgbtRed = 255;
				rgb_out_e[Y][X].rgbtGreen = 255;
			}
			else
			{
				rgb_out_e[Y][X].rgbtRed = 255;
			}
		}
	}

	string sm2 = "M2", su = "U", se = "E", sr = "R";

	BMPWrite(rgb_out_m2, imWidth, imHeight, "c:\\temp\\texturefeatures_OMP" + sm2 + ".bmp");

	BMPWrite(rgb_out_u, imWidth, imHeight, "c:\\temp\\texturefeatures_OMP" + su + ".bmp");

	BMPWrite(rgb_out_e, imWidth, imHeight, "c:\\temp\\texturefeatures_OMP" + se + ".bmp");

	BMPWrite(rgb_out_r, imWidth, imHeight, "c:\\temp\\texturefeatures_OMP" + sr + ".bmp");

	for (int i = 0; i < imHeight; i++)
	{
		delete[] bimage[i];
		delete[] M2[i];
		delete[] R[i];
		delete[] U[i];
		delete[] E[i];
	}

	double t_end = omp_get_wtime();

	delete[] rgb_out_m2[0]; delete[] rgb_out_u[0]; delete[] rgb_out_e[0]; delete[] rgb_out_r[0];
	delete[] rgb_out_m2; delete[] rgb_out_u; delete[] rgb_out_e; delete[] rgb_out_r;
	delete[] M2; delete[] R; delete[] E; delete[] U; delete[] bimage;
	return (t_end - t_start) * 1000;
}

// Функция рассчёта текстурных признаков с использованием tbb::parallel_for
double texturefeaturesTBB(RGBTRIPLE**& rgb_in, int &imWidth, int &imHeight, int &RH, int &RW)
{
	double t_start = omp_get_wtime();
	int size = (RH * 2 + 1) * (RW * 2 + 1);

	RGBTRIPLE** rgb_out_m2 = new RGBTRIPLE * [imHeight], ** rgb_out_r = new RGBTRIPLE * [imHeight],
		** rgb_out_u = new RGBTRIPLE * [imHeight], ** rgb_out_e = new RGBTRIPLE * [imHeight];

	// выделение памяти под новое (выходное) изображение (rgb_out)
	rgb_out_m2[0] = new RGBTRIPLE[imWidth * imHeight];
	rgb_out_u[0] = new RGBTRIPLE[imWidth * imHeight];
	rgb_out_r[0] = new RGBTRIPLE[imWidth * imHeight];
	rgb_out_e[0] = new RGBTRIPLE[imWidth * imHeight];

	for (int i = 1; i < imHeight; i++)
	{
		rgb_out_m2[i] = &rgb_out_m2[0][imWidth * i];
		rgb_out_u[i] = &rgb_out_u[0][imWidth * i];
		rgb_out_r[i] = &rgb_out_r[0][imWidth * i];
		rgb_out_e[i] = &rgb_out_e[0][imWidth * i];
	}

	int** bimage = new int* [imHeight];
	double** M2 = new double* [imHeight],
		** R = new double* [imHeight], ** U = new double* [imHeight],
		** E = new double* [imHeight];

	for (int i = 0; i < imHeight; i++)
	{
		bimage[i] = new int[imWidth];
		M2[i] = new double[imWidth];
		R[i] = new double[imWidth];
		U[i] = new double[imWidth];
		E[i] = new double[imWidth];
	}

	tbb::parallel_for(tbb::blocked_range2d<int>(0, imHeight, 0, imWidth), [&](tbb::blocked_range2d<int> r)
		{
			for (int Y = r.rows().begin(); Y < r.rows().end(); Y++)
				for (int X = r.cols().begin(); X < r.cols().end(); X++)
					bimage[Y][X] = rgb_in[Y][X].rgbtRed * 0.299 + rgb_in[Y][X].rgbtGreen * 0.587 + rgb_in[Y][X].rgbtBlue * 0.114;
		});

	double max_r = 0, max_m2 = 0, max_e = 0, max_u = 0, min_m2 = 0, min_u = 0, min_e = 0, min_r = 0;

	tbb::parallel_for(tbb::blocked_range2d<int>(0, imHeight, 0, imWidth), [&](tbb::blocked_range2d<int> r)
		{
			double hist[256];

			for (int Y = r.rows().begin(); Y < r.rows().end(); Y++)
			for (int X = r.cols().begin(); X < r.cols().end(); X++)
			{
				for (int i = 0; i <= 255; i++)
					hist[i] = 0;

				for (int DY = -RH; DY <= RH; DY++)
				{
					int KY = Y + DY;
					if (KY < 0)
						KY = 0;
					if (KY > imHeight - 1)
						KY = imHeight - 1;
					for (int DX = -RW; DX <= RW; DX++)
					{
						int KX = X + DX;
						if (KX < 0)
							KX = 0;
						if (KX > imWidth - 1)
							KX = imWidth - 1;
						hist[bimage[KY][KX]] = hist[bimage[KY][KX]] + 1;
					}
				}

				int Size = (RH * 2 + 1) * (RW * 2 + 1);
				double m = 0, m2 = 0, e = 0, u = 0;

				for (int i = 0; i <= 255; i++)
					hist[i] = hist[i] / size;
				for (int i = 0; i <= 255; i++)
					m += hist[i] * i;
				for (int i = 0; i <= 255; i++)
				{
					m2 += pow(i - m, 2) * hist[i];
					if (hist[i] != 0)
						e += hist[i] * log2(hist[i]);
					u += pow(hist[i], 2);
				}
				M2[Y][X] = m2;
				U[Y][X] = u;
				R[Y][X] = 1 - (1 / (1 + m2));
				E[Y][X] = -1 * e;

				if (Y == 0 && X == 0)
				{
					max_m2 = m2;
					max_u = u;
					max_r = (1 - (1 / (1 + m2)));
					max_e = (-1 * e);
					min_m2 = m2;
					min_u = u;
					min_r = (1 - (1 / (1 + m2)));
					min_e = (-1 * e);
				}
				else
				{
					if (M2[Y][X] > max_m2)
						max_m2 = M2[Y][X];
					if (R[Y][X] > max_r)
						max_r = R[Y][X];
					if (U[Y][X] > max_u)
						max_u = U[Y][X];
					if (E[Y][X] > max_e)
						max_e = E[Y][X];

					if (m2 < min_m2)
						min_m2 = m2;
					if (R[Y][X] < min_r)
						min_r = R[Y][X];
					if (u < min_u)
						min_u = u;
					if (-1 * e < min_e)
						min_e = -1 * e;
				}
			}
		});
	double T1_m2 = (max_m2 - min_m2) * 0.3 + min_m2;
	double T2_m2 = (max_m2 - min_m2) * 0.7 + min_m2;

	double T1_u = (max_u - min_u) * 0.3 + min_u;
	double T2_u = (max_u - min_u) * 0.7 + min_u;

	double T1_e = (max_e - min_e) * 0.3 + min_e;
	double T2_e = (max_e - min_e) * 0.7 + min_e;

	double T1_r = (max_r - min_r) * 0.3 + min_r;
	double T2_r = (max_r - min_r) * 0.7 + min_r;

	tbb::parallel_for(tbb::blocked_range2d<int>(0, imHeight, 0, imWidth), [&](tbb::blocked_range2d<int> r)
		{
			for (int Y = r.rows().begin(); Y < r.rows().end(); Y++)
			for (int X = r.cols().begin(); X < r.cols().end(); X++)
			{
				if (M2[Y][X] <= T1_m2)
				{
					rgb_out_m2[Y][X].rgbtGreen = 255;
				}
				else if (T1_m2 < M2[Y][X] && M2[Y][X] < T2_m2)
				{
					rgb_out_m2[Y][X].rgbtRed = 255;
					rgb_out_m2[Y][X].rgbtGreen = 255;
				}
				else
				{
					rgb_out_m2[Y][X].rgbtRed = 255;
				}

				if (U[Y][X] <= T1_u)
				{
					rgb_out_u[Y][X].rgbtGreen = 255;
				}
				else if (T1_u < U[Y][X] && U[Y][X] < T2_u)
				{
					rgb_out_u[Y][X].rgbtRed = 255;
					rgb_out_u[Y][X].rgbtGreen = 255;
				}
				else
				{
					rgb_out_u[Y][X].rgbtRed = 255;
				}

				if (R[Y][X] <= T1_r)
				{
					rgb_out_r[Y][X].rgbtGreen = 255;
				}
				else if (T1_r < R[Y][X] && R[Y][X] < T2_r)
				{
					rgb_out_r[Y][X].rgbtRed = 255;
					rgb_out_r[Y][X].rgbtGreen = 255;
				}
				else
				{
					rgb_out_r[Y][X].rgbtRed = 255;
				}

				if (E[Y][X] <= T1_e)
				{
					rgb_out_e[Y][X].rgbtGreen = 255;
				}
				else if (T1_e < E[Y][X] && E[Y][X] < T2_e)
				{
					rgb_out_e[Y][X].rgbtRed = 255;
					rgb_out_e[Y][X].rgbtGreen = 255;
				}
				else
				{
					rgb_out_e[Y][X].rgbtRed = 255;
				}
			}
		});

	string sm2 = "M2", su = "U", se = "E", sr = "R";

	BMPWrite(rgb_out_m2, imWidth, imHeight, "c:\\temp\\texturefeatures_TBB" + sm2 + ".bmp");

	BMPWrite(rgb_out_u, imWidth, imHeight, "c:\\temp\\texturefeatures_TBB" + su + ".bmp");

	BMPWrite(rgb_out_e, imWidth, imHeight, "c:\\temp\\texturefeatures_TBB" + se + ".bmp");

	BMPWrite(rgb_out_r, imWidth, imHeight, "c:\\temp\\texturefeatures_TBB" + sr + ".bmp");

	for (int i = 0; i < imHeight; i++)
	{
		delete[] bimage[i];
		delete[] M2[i];
		delete[] R[i];
		delete[] U[i];
		delete[] E[i];
	}

	double t_end = omp_get_wtime();

	delete[] rgb_out_m2[0]; delete[] rgb_out_u[0]; delete[] rgb_out_e[0]; delete[] rgb_out_r[0];
	delete[] rgb_out_m2; delete[] rgb_out_u; delete[] rgb_out_e; delete[] rgb_out_r;
	delete[] M2; delete[] R; delete[] E; delete[] U; delete[] bimage;
	return (t_end - t_start) * 1000;
}

// Быстрая линейная фильтрация Среднеарифметическим фильтром с использованием omp parallel for
double QuickLinearArithmeticMeanFilter(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int imWidth, int  imHeight, int RH, int RW)
{
	double time_start = omp_get_wtime();
	
	vector <int> Summ_Values_R, Summ_Values_G, Summ_Values_B;

	int size = (RH * 2 + 1) * (RW * 2 + 1);
#pragma omp parallel for private (Summ_Values_R, Summ_Values_G, Summ_Values_B)
	for (int Y = 0; Y < imHeight; Y++)
	{
		for (int X = 0; X < imWidth; X++)
		{
			int Summ_Value_R = 0, Summ_Value_G = 0, Summ_Value_B = 0;

			if (X == 0)
			{
				for (int DY = -RH; DY <= RH; DY++)
				{
					int KY = Y + DY;
					if (KY < 0)
						KY = 0;
					if (KY > imHeight - 1)
						KY = imHeight - 1;
					for (int DX = -RW; DX <= RW; DX++)
					{
						int KX = X + DX;
						if (KX < 0)
							KX = 0;
						if (KX > imWidth - 1)
							KX = imWidth - 1;
						Summ_Value_R += rgb_in[KY][KX].rgbtRed;
						Summ_Value_G += rgb_in[KY][KX].rgbtGreen;
						Summ_Value_B += rgb_in[KY][KX].rgbtBlue;
					}
				}
				rgb_out[Y][X].rgbtRed = Summ_Value_R / size;
				rgb_out[Y][X].rgbtGreen = Summ_Value_G / size;
				rgb_out[Y][X].rgbtBlue = Summ_Value_B / size;
				
				Summ_Values_R.push_back(Summ_Value_R);
				Summ_Values_G.push_back(Summ_Value_G);
				Summ_Values_B.push_back(Summ_Value_B);
			}
			else
			{
				int KX1 = X - RW - 1;
				if (KX1 < 0)
					KX1 = 0;
				int KX2 = X + RW;
				if (KX2 > imWidth - 1)
					KX2 = imWidth - 1;
				for (int DY = -RH; DY <= RH; DY++)
				{
					int KY = Y + DY;
					if (KY < 0)
						KY = 0;
					if (KY > imHeight - 1)
						KY = imHeight - 1;
					Summ_Value_R -= rgb_in[KY][KX1].rgbtRed;
					Summ_Value_R += rgb_in[KY][KX2].rgbtRed;

					Summ_Value_G -= rgb_in[KY][KX1].rgbtGreen;
					Summ_Value_G += rgb_in[KY][KX2].rgbtGreen;

					Summ_Value_B -= rgb_in[KY][KX1].rgbtBlue;
					Summ_Value_B += rgb_in[KY][KX2].rgbtBlue;
				}
				Summ_Value_R += Summ_Values_R[X - 1];
				Summ_Value_G += Summ_Values_G[X - 1];
				Summ_Value_B += Summ_Values_B[X - 1];

				rgb_out[Y][X].rgbtRed = Summ_Value_R / size;
				rgb_out[Y][X].rgbtGreen = Summ_Value_G / size;
				rgb_out[Y][X].rgbtBlue = Summ_Value_B / size;

				Summ_Values_R.push_back(Summ_Value_R);
				Summ_Values_G.push_back(Summ_Value_G);
				Summ_Values_B.push_back(Summ_Value_B);
			}
		}
		Summ_Values_R.clear();
		Summ_Values_G.clear();
		Summ_Values_B.clear();
	}

	return (omp_get_wtime() - time_start) * 1000;
}

void LinearGaussFilterSection(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int stY, int imWidth, int  imHeight, int RH, int RW, double**& MATR_coef)
{
	for (int Y = stY; Y < imHeight; Y++)
	{
		for (int X = 0; X < imWidth; X++)
		{
			int LinF_Value_R = 0, LinF_Value_G = 0, LinF_Value_B = 0;
			for (int DY = -RH; DY <= RH; DY++)
			{
				int KY = Y + DY;
				if (KY < 0)
					KY = 0;
				if (KY > imHeight - 1)
					KY = imHeight - 1;
				for (int DX = -RW; DX <= RW; DX++)
				{
					int KX = X + DX;
					if (KX < 0)
						KX = 0;
					if (KX > imWidth - 1)
						KX = imWidth - 1;
					LinF_Value_R += rgb_in[KY][KX].rgbtRed * MATR_coef[DY + RH][DX + RW];
					LinF_Value_G += rgb_in[KY][KX].rgbtGreen * MATR_coef[DY + RH][DX + RW];
					LinF_Value_B += rgb_in[KY][KX].rgbtBlue * MATR_coef[DY + RH][DX + RW];
				}
			}
			vector<int> LinF_Values{ LinF_Value_R , LinF_Value_G ,LinF_Value_B };
			for (auto it : LinF_Values)
			{
				if (it < 0)
					it = 0;
				if (it > 255)
					it = 255;
			}
			rgb_out[Y][X].rgbtRed = LinF_Value_R;
			rgb_out[Y][X].rgbtGreen = LinF_Value_G;
			rgb_out[Y][X].rgbtBlue = LinF_Value_B;
		}
	}
}

// Линейная фильтрация методом Гаусса с использованием omp parallel sections
double LinearGaussFilterSections(RGBTRIPLE**& rgb_in, RGBTRIPLE**& rgb_out, int imWidth, int  imHeight, int RH, int RW)
{
	double time_start = omp_get_wtime();

	int M = 2 * RH + 1, N = 2 * RW + 1;
	double** MATR_coef = new double* [M];
	MATR_coef[0] = new double[M * N];
	for (int i = 1; i < M; i++)
	{
		MATR_coef[i] = &MATR_coef[0][N * i];
	}

	fillingMatrCoef(MATR_coef, RH, RW, RH);

	int n_t = omp_get_max_threads();
	int st = 0;
	int s1 = imHeight / n_t;

	int s2 = imHeight * 2 / n_t;
	int s3 = imHeight * 3 / n_t;
	int se = imHeight;

#pragma omp parallel sections
	{
#pragma omp section
		{
			if (n_t > 0)
				LinearGaussFilterSection(rgb_in, rgb_out, st, imWidth, s1, RH, RW, MATR_coef);
		}
#pragma omp section
		{
			if (n_t > 1)
				LinearGaussFilterSection(rgb_in, rgb_out, s1, imWidth, s2, RH, RW, MATR_coef);
		}
#pragma omp section
		{
			if (n_t > 2)
				LinearGaussFilterSection(rgb_in, rgb_out, s2, imWidth, s3, RH, RW, MATR_coef);
		}
#pragma omp section
		{
			if (n_t > 3)
				LinearGaussFilterSection(rgb_in, rgb_out, s3, imWidth, se, RH, RW, MATR_coef);
		}
	}

	return (omp_get_wtime() - time_start) * 1000;
}

void LinearArithmeticMeanFilterSection(RGBTRIPLE** rgb_in, RGBTRIPLE** rgb_out, int stY, int imWidth, int  imHeight, int RH, int RW)
{
	int size = (RH * 2 + 1) * (RW * 2 + 1);

	for (int Y = stY; Y < imHeight; Y++)
	{
		for (int X = 0; X < imWidth; X++)
		{
			int Summ_Value_R = 0, Summ_Value_G = 0, Summ_Value_B = 0;

			for (int DY = -RH; DY <= RH; DY++)
			{
				int KY = Y + DY;
				if (KY < 0)
					KY = 0;
				if (KY > imHeight - 1)
					KY = imHeight - 1;
				for (int DX = -RW; DX <= RW; DX++)
				{
					int KX = X + DX;
					if (KX < 0)
						KX = 0;
					if (KX > imWidth - 1)
						KX = imWidth - 1;
					Summ_Value_R += rgb_in[KY][KX].rgbtRed;
					Summ_Value_G += rgb_in[KY][KX].rgbtGreen;
					Summ_Value_B += rgb_in[KY][KX].rgbtBlue;
				}
			}
			rgb_out[Y][X].rgbtRed = Summ_Value_R / size;
			rgb_out[Y][X].rgbtGreen = Summ_Value_G / size;
			rgb_out[Y][X].rgbtBlue = Summ_Value_B / size;
		}
	}
}

// Линейная фильтрация Среднеарифметическим фильтром с использованием omp parallel sections
double LinearArithmeticMeanFilterSections(RGBTRIPLE** rgb_in, RGBTRIPLE** rgb_out, int imWidth, int  imHeight, int RH, int RW)
{
	double time_start = omp_get_wtime();

	int n_t = omp_get_max_threads();
	int st = 0;
	int s1 = imHeight / n_t;

	int s2 = imHeight * 2 / n_t;
	int s3 = imHeight * 3 / n_t;
	int se = imHeight;

#pragma omp parallel sections
	{
#pragma omp section
		{
				LinearArithmeticMeanFilterSection(rgb_in, rgb_out, st, imWidth, s1, RH, RW);
		}
#pragma omp section
		{
			if (n_t > 1)
				LinearArithmeticMeanFilterSection(rgb_in, rgb_out, s1, imWidth, s2, RH, RW);
		}
#pragma omp section
		{
			if (n_t > 2)
				LinearArithmeticMeanFilterSection(rgb_in, rgb_out, s2, imWidth, s3, RH, RW);
		}
#pragma omp section
		{
			if (n_t > 3)
				LinearArithmeticMeanFilterSection(rgb_in, rgb_out, s3, imWidth, se, RH, RW);
		}
	}
	
	return (omp_get_wtime() - time_start) * 1000;
}

void QuickLinearArithmeticMeanFilterSection(RGBTRIPLE** rgb_in, RGBTRIPLE** rgb_out, int stY, int imWidth, int  imHeight, int RH, int RW)
{

	vector <int> Summ_Values_R, Summ_Values_G, Summ_Values_B;

	int size = (RH * 2 + 1) * (RW * 2 + 1);

	for (int Y = stY; Y < imHeight; Y++)
	{
		for (int X = 0; X < imWidth; X++)
		{
			int Summ_Value_R = 0, Summ_Value_G = 0, Summ_Value_B = 0;

			if (X == 0)
			{
				for (int DY = -RH; DY <= RH; DY++)
				{
					int KY = Y + DY;
					if (KY < 0)
						KY = 0;
					if (KY > imHeight - 1)
						KY = imHeight - 1;
					for (int DX = -RW; DX <= RW; DX++)
					{
						int KX = X + DX;
						if (KX < 0)
							KX = 0;
						if (KX > imWidth - 1)
							KX = imWidth - 1;
						Summ_Value_R += rgb_in[KY][KX].rgbtRed;
						Summ_Value_G += rgb_in[KY][KX].rgbtGreen;
						Summ_Value_B += rgb_in[KY][KX].rgbtBlue;
					}
				}
				rgb_out[Y][X].rgbtRed = Summ_Value_R / size;
				rgb_out[Y][X].rgbtGreen = Summ_Value_G / size;
				rgb_out[Y][X].rgbtBlue = Summ_Value_B / size;

				Summ_Values_R.push_back(Summ_Value_R);
				Summ_Values_G.push_back(Summ_Value_G);
				Summ_Values_B.push_back(Summ_Value_B);
			}
			else
			{
				int KX1 = X - RW - 1;
				if (KX1 < 0)
					KX1 = 0;
				int KX2 = X + RW;
				if (KX2 > imWidth - 1)
					KX2 = imWidth - 1;
				for (int DY = -RH; DY <= RH; DY++)
				{
					int KY = Y + DY;
					if (KY < 0)
						KY = 0;
					if (KY > imHeight - 1)
						KY = imHeight - 1;
					Summ_Value_R -= rgb_in[KY][KX1].rgbtRed;
					Summ_Value_R += rgb_in[KY][KX2].rgbtRed;

					Summ_Value_G -= rgb_in[KY][KX1].rgbtGreen;
					Summ_Value_G += rgb_in[KY][KX2].rgbtGreen;

					Summ_Value_B -= rgb_in[KY][KX1].rgbtBlue;
					Summ_Value_B += rgb_in[KY][KX2].rgbtBlue;
				}
				Summ_Value_R += Summ_Values_R[X - 1];
				Summ_Value_G += Summ_Values_G[X - 1];
				Summ_Value_B += Summ_Values_B[X - 1];

				rgb_out[Y][X].rgbtRed = Summ_Value_R / size;
				rgb_out[Y][X].rgbtGreen = Summ_Value_G / size;
				rgb_out[Y][X].rgbtBlue = Summ_Value_B / size;

				Summ_Values_R.push_back(Summ_Value_R);
				Summ_Values_G.push_back(Summ_Value_G);
				Summ_Values_B.push_back(Summ_Value_B);
			}
		}
		Summ_Values_R.clear();
		Summ_Values_G.clear();
		Summ_Values_B.clear();
	}

}

// Быстрая линейная фильтрация Среднеарифметическим фильтром с использованием omp parallel sections
double QuickLinearArithmeticMeanFilterSections(RGBTRIPLE** rgb_in, RGBTRIPLE** rgb_out, int imWidth, int  imHeight, int RH, int RW)
{
	double time_start = omp_get_wtime();

	int n_t = omp_get_max_threads();
	int st = 0;
	int s1 = imHeight / n_t;

	int s2 = imHeight * 2 / n_t;
	int s3 = imHeight * 3 / n_t;
	int se = imHeight;

#pragma omp parallel sections
	{
#pragma omp section
		{
				QuickLinearArithmeticMeanFilterSection(rgb_in, rgb_out, st, imWidth, s1, RH, RW);
		}
#pragma omp section
		{
			if (n_t > 1)
				QuickLinearArithmeticMeanFilterSection(rgb_in, rgb_out, s1, imWidth, s2, RH, RW);
		}
#pragma omp section
		{
			if (n_t > 2)
				QuickLinearArithmeticMeanFilterSection(rgb_in, rgb_out, s2, imWidth, s3, RH, RW);
		}
#pragma omp section
		{
			if (n_t > 3)
				QuickLinearArithmeticMeanFilterSection(rgb_in, rgb_out, s3, imWidth, se, RH, RW);
		}
	}

	return (omp_get_wtime() - time_start) * 1000;
}

// Функция рассчёта доверительного интервала
double AvgTrustedIntervalAVG(double*& times, int cnt)
{
	// вычисление среднеарифметического значения
	double avg = 0;
	for (int i = 0; i < cnt; i++)
	{
		// подсчет в переменную суммы
		avg += times[i];
	}
	// деление на количество
	avg /= cnt;
	// подсчет стандартного отклонения
	double sd = 0, newAVg = 0;
	int newCnt = 0;
	for (int i = 0; i < cnt; i++)
	{
		sd += (times[i] - avg) * (times[i] - avg);
	}
	sd /= (cnt - 1.0);
	sd = sqrt(sd);
	// вычисление нового среднего значения в доверительном интервале
	// с использованием среднеарифметического значения
	//
	for (int i = 0; i < cnt; i++)
	{
		if (avg - sd <= times[i] && times[i] <= avg + sd)
		{
			newAVg += times[i];
			newCnt++;
		}
	}
	if (newCnt == 0) newCnt = 1;
	return newAVg / newCnt;
}

int main()
{
	SetConsoleCP(1251);
	SetConsoleOutputCP(1251);
	double** times = new double* [10];

	for (int j = 0; j < 10; j++)
		times[j] = new double[ExpCount];

	// Для 4 изображений
	for (int dataset = 0; dataset <= 4; dataset++)
	{
		stringstream ss1;
		ss1 << dataset;

		RGBTRIPLE** rgb_in = nullptr, ** rgb_out = nullptr;
		BITMAPFILEHEADER header;
		BITMAPINFOHEADER bmiHeader;
		int imWidth = 0, imHeight = 0;

		// чтение BMP файла с прописанным путем
		BMPRead(rgb_in, header, bmiHeader, "c:\\temp\\sample" + ss1.str() + ".bmp");

		// считываем из заголовочной области параметры изображения
		imWidth = bmiHeader.biWidth;
		imHeight = bmiHeader.biHeight;

		// Вывод на экран параметров изображение (ширина х вымота)
		std::cout << "\nImage params:" << imWidth << "x" << imHeight << endl;

		// выделение памяти под новое (выходное) изображение (rgb_out)
		rgb_out = new RGBTRIPLE * [imHeight];
		rgb_out[0] = new RGBTRIPLE[imWidth * imHeight];

		for (int i = 1; i < imHeight; i++)
		{
			rgb_out[i] = &rgb_out[0][imWidth * i];
		}

		for (int threads = 4; threads <= 4; threads++)
		{
			omp_set_num_threads(threads);

			// Установка количества потоков
			tbb::global_control
				global_limit(tbb::global_control::max_allowed_parallelism, threads);

			std::cout << "\nТекущее число потоков " << omp_get_max_threads() << endl;

			stringstream ss2;
			ss2 << omp_get_max_threads();

			for (int RH = 3, RW = 3; RH <= 7; RH += 2, RW += 2)
			{
				for (int i = 0; i < ExpCount; i++)
				{
					stringstream ss3;
					ss3 << RW;
					times[0][i] = LinearGaussFilter(rgb_in, rgb_out, imWidth,imHeight, RH, RW);
					BMPWrite(rgb_out, imWidth, imHeight, "c:\\temp\\sample" + ss1.str() + "RHRW" + ss3.str() + "_" + ss2.str() + "_threads_LinearGaussFilter.bmp");

					times[1][i] = LinearArithmeticMeanFilter(rgb_in, rgb_out, imWidth, imHeight, RH, RW);
					BMPWrite(rgb_out, imWidth, imHeight, "c:\\temp\\sample" + ss1.str() + "RHRW" + ss3.str() + "_" + ss2.str() + "_threads_LinearArithmeticMeanFilter.bmp");

					times[2][i] = QuickLinearArithmeticMeanFilter(rgb_in, rgb_out, imWidth, imHeight, RH, RW);
					BMPWrite(rgb_out, imWidth, imHeight, "c:\\temp\\sample" + ss1.str() + "RHRW" + ss3.str() + "_" + ss2.str() + "_threads_QuickLinearArithmeticMeanFilter.bmp");

					times[3][i] = LinearGaussFilterSections(rgb_in, rgb_out, imWidth, imHeight, RH, RW);
					BMPWrite(rgb_out, imWidth, imHeight, "c:\\temp\\sample" + ss1.str() + "RHRW" + ss3.str() + "_" + ss2.str() + "_threads_LinearGaussFilterSections.bmp");

					times[4][i] = LinearArithmeticMeanFilterSections(rgb_in, rgb_out, imWidth, imHeight, RH, RW);
					BMPWrite(rgb_out, imWidth, imHeight, "c:\\temp\\sample" + ss1.str() + "RHRW" + ss3.str() + "_" + ss2.str() + "_threads_LinearArithmeticMeanFilterSections.bmp");

					times[5][i] = QuickLinearArithmeticMeanFilterSections(rgb_in, rgb_out, imWidth, imHeight, RH, RW);
					BMPWrite(rgb_out, imWidth, imHeight, "c:\\temp\\sample" + ss1.str() + "RHRW" + ss3.str() + "_" + ss2.str() + "_threads_QuickLinearArithmeticMeanFilterSection.bmp");

					times[6][i] = LinearGaussFilterTBB(rgb_in, rgb_out, imWidth,imHeight, RH, RW);
					BMPWrite(rgb_out, imWidth, imHeight, "c:\\temp\\sample" + ss1.str() + "RHRW" + ss3.str() + "_" + ss2.str() + "_threads_LinearGaussFilterTBB.bmp");

					times[7][i] = LinearArithmeticMeanFilterTBB(rgb_in, rgb_out, imWidth, imHeight, RH, RW);
					BMPWrite(rgb_out, imWidth, imHeight, "c:\\temp\\sample" + ss1.str() + "RHRW" + ss3.str() + "_" + ss2.str() + "_threads_LinearArithmeticMeanFilterTBB.bmp");

					times[8][i] = texturefeatures(rgb_in, imWidth, imHeight, RH, RW);

					times[9][i] = texturefeaturesTBB(rgb_in, imWidth, imHeight, RH, RW);
				}

				std::cout << "\nДля RH = RW = " << RH << " время (TBB) Гаусса составило " << AvgTrustedIntervalAVG(times[6], ExpCount) << endl;

				std::cout << "\nДля RH = RW = " << RH << " время (TBB) Среднеарифметического фильтра составило " << AvgTrustedIntervalAVG(times[7], ExpCount) << endl;

				std::cout << "\nДля RH = RW = " << RH << " время (Parallel) текстурных признаков составило " << AvgTrustedIntervalAVG(times[8], ExpCount) << endl;

				std::cout << "\nДля RH = RW = " << RH << " время (TBB) текстурных признаков составило " << AvgTrustedIntervalAVG(times[9], ExpCount) << endl;
			}
		}
	}

	return 0;
}
