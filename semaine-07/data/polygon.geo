lc = 0.5;

Point(1) = {0, 0, 0, lc};
Point(2) = {170, 0, 0, lc};
Point(3) = {170, 20, 0, lc};
Point(4) = {140, 20, 0, lc};
Point(5) = {139, 19, 0, lc};
Point(6) = {131, 19, 0, lc};
Point(7) = {130, 20, 0, lc};
Point(8) = {120, 20, 0, lc};
Point(9) = {60, 50, 0, lc};
Point(10) = {40, 50, 0, lc};
Point(11) = {39, 49, 0, lc};
Point(12) = {21, 49, 0, lc};
Point(13) = {20, 50, 0, lc};
Point(14) = {0, 50, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 7};
Line(7) = {7, 8};
Line(8) = {8, 9};
Line(9) = {9, 10};
Line(10) = {10, 11};
Line(11) = {11, 12};
Line(12) = {12, 13};
Line(13) = {13, 14};
Line(14) = {14, 1};

Line Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
Plane Surface(1) = {1};