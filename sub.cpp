#include<opencv2\core\core.hpp>
#include<opencv2\highgui\highgui.hpp>
#include<opencv2\imgproc\imgproc.hpp>
#include<stdio.h>
#include<iostream>
#include<math.h>

using namespace cv;
using namespace std;

int main()
{
	namedWindow("Source", CV_WINDOW_AUTOSIZE);

	string pathToData("C:/office/input/in%06d.jpg");	//add the string to pathToData

	VideoCapture sequence(pathToData);					//sequence points to the path
	Vector<Mat> images;
	Mat src;
	for (;;)
	{
		sequence >> src;								//add the images one by one in src
		if (src.empty())								//check if the image series exists
		{
			cout << "End of Sequence" << endl;			
			break;
		}
		//imshow("Source", src);
		//waitKey(10);
		images.push_back(src.clone());					//push back the cloned value of src into images
	}
	float size = images.size();
	float p = 0.000001f;								// initialize the value of p
	for (int i = 0; i < size; i++)
	{	
		images[i].convertTo(images[i], CV_32FC1);		//convert to 32 bit single channel
		cvtColor(images[i], images[i], CV_BGR2GRAY, 1);	// convert BGR to gray
		blur(images[i], images[i], Size(3, 3));			// blur the images (remove noise)
	}
	
	//cout<<images[0].rows<< "\t" << images[0].cols<<endl;
	//waitKey(0);

	Mat q, meant, meant_sq, sigma_sq;
	blur(images[0], meant, Size(3, 3));					// initial mean
	//cout << meant << endl;
	blur(images[0].mul(images[0]), meant_sq, Size(3, 3));		// initial mean_sq
	q = meant.mul(meant);
	subtract(meant_sq, q, sigma_sq);						//initial variance
	//cout << sigma_sq << endl;
	//waitKey(0);

	Mat d, d_sq, x, y, z, sigma, k1, kout;
	Mat k = Mat(images[0].rows, images[0].cols, CV_32FC1, float(30.0));			//initialize k

	int j = images[0].rows;
	int l = images[0].cols;

	for (int a = 1; a < size; a++)
	{
		x = p*images[a];
		z = (1 - p)*meant;
		add(x, z, meant);							//update mean
		//cout << meant << endl;
		absdiff(images[a], meant, d);				// value of d (Euclidian distance)
		//cout << d << endl;
		d_sq = d.mul(d);
		//cout << d_sq<<endl;
		x = (1 - p)*sigma_sq;
		y = d_sq*p;			
		add(x, y, sigma_sq);						//update variance						
		//cout << sigma_sq << endl;
		sqrt(sigma_sq, sigma);
		//cout << sigma << endl;

		k1 = d / sigma;								//value to be compared with the threshold
		//cout << k1 << endl;

		compare(k1, k, images[a], CMP_GT);			//compare and accordingly update the pixel in images[]
		imshow("Source", images[a]);				//output the image sequence
		waitKey(1);
	}

	imshow("Source", images[619]);					//output "in000620.jpg
	imwrite("banthia_output.jpg", images[619]);		//save it
	waitKey(0);
	return 0;
}
