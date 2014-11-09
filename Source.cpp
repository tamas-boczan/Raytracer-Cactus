//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2014-tol.          
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk. 
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat. 
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni (printf is fajlmuvelet!)
// - new operatort hivni az onInitialization függvényt kivéve, a lefoglalt adat korrekt felszabadítása nélkül 
// - felesleges programsorokat a beadott programban hagyni
// - tovabbi kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan gl/glu/glut fuggvenyek hasznalhatok, amelyek
// 1. Az oran a feladatkiadasig elhangzottak ES (logikai AND muvelet)
// 2. Az alabbi listaban szerepelnek:  
// Rendering pass: glBegin, glVertex[2|3]f, glColor3f, glNormal3f, glTexCoord2f, glEnd, glDrawPixels
// Transzformaciok: glViewport, glMatrixMode, glLoadIdentity, glMultMatrixf, gluOrtho2D, 
// glTranslatef, glRotatef, glScalef, gluLookAt, gluPerspective, glPushMatrix, glPopMatrix,
// Illuminacio: glMaterialfv, glMaterialfv, glMaterialf, glLightfv
// Texturazas: glGenTextures, glBindTexture, glTexParameteri, glTexImage2D, glTexEnvi, 
// Pipeline vezerles: glShadeModel, glEnable/Disable a kovetkezokre:
// GL_LIGHTING, GL_NORMALIZE, GL_DEPTH_TEST, GL_CULL_FACE, GL_TEXTURE_2D, GL_BLEND, GL_LIGHT[0..7]
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Boczán Tamás
// Neptun : A5X61F
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy 
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem. 
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a 
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb 
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem, 
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.  
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat 
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES

#include <math.h>
#include <stdlib.h>

#if defined(__APPLE__)                                                                                                                                                                                                            
#include <OpenGL/gl.h>                                                                                                                                                                                                            
#include <OpenGL/glu.h>                                                                                                                                                                                                           
#include <GLUT/glut.h>                                                                                                                                                                                                            
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)                                                                                                                                                                       
#include <windows.h>                                                                                                                                                                                                              
#endif

#include <GL/gl.h>                                                                                                                                                                                                                
#include <GL/glu.h>                                                                                                                                                                                                               
#include <GL/glut.h>                                                                                                                                                                                                              

#endif


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Innentol modosithatod...

#define FLT_MAX 10000.0f
#define NEAR_ZERO 0.001f
#define PI 3.14159265359f

//--------------------------------------------------------
// 3D Vektor
//--------------------------------------------------------
struct Vector {
    float x, y, z;

    Vector() {
        x = y = z = 0;
    }

    Vector(float x0, float y0, float z0 = 0) {
        x = x0;
        y = y0;
        z = z0;
    }

    Vector operator*(float a) const {
        return Vector(x * a, y * a, z * a);
    }

    Vector operator/(float a) const {
        return Vector(x / a, y / a, z / a);
    }

    Vector operator+(const Vector &v) const {
        return Vector(x + v.x, y + v.y, z + v.z);
    }

    Vector operator-(const Vector &v) const {
        return Vector(x - v.x, y - v.y, z - v.z);
    }

    float operator*(const Vector &v) const {    // dot product
        return (x * v.x + y * v.y + z * v.z);
    }

    Vector operator%(const Vector &v) const {    // cross product
        return Vector(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }

    float length() const {
        return sqrtf(x * x + y * y + z * z);
    }

    Vector normalized() const {
        return *this / length();
    }

    Vector operator*=(float a) {
        x *= a;
        y *= a;
        z *= a;
        return *this;
    }
};

//--------------------------------------------------------
// Spektrum illetve szin
//--------------------------------------------------------
struct Color {
    float r, g, b;

    Color() {
        r = g = b = 0;
    }

    Color(float r0, float g0, float b0) {
        r = r0;
        g = g0;
        b = b0;
    }

    Color operator*(float a) const {
        return Color(r * a, g * a, b * a);
    }

    Color operator/(float a) const {
        return Color(r / a, g / a, b / a);
    }

    Color operator*(const Color &c) const {
        return Color(r * c.r, g * c.g, b * c.b);
    }

    Color operator/(const Color &c) const {
        return Color(r / c.r, g / c.g, b / c.b);
    }

    Color operator+(const Color &c) const {
        return Color(r + c.r, g + c.g, b + c.b);
    }

    Color operator-(const Color &c) const {
        return Color(r - c.r, g - c.g, b - c.b);
    }

    Color operator+=(const Color &c) {
        r += c.r;
        g += c.g;
        b += c.b;
        return *this;
    }

    Color operator*=(const Color &c) {
        r *= c.r;
        g *= c.g;
        b *= c.b;
        return *this;
    }
};

class Matrix4D {
    float mx[4][4];
    size_t rowsFilled;
    size_t columnsFilled;

public:
    Matrix4D() {
        allZero();
        rowsFilled = 0;
        columnsFilled = 0;
    }

    Matrix4D(
            float f00, float f01, float f02, float f03,
            float f10, float f11, float f12, float f13,
            float f20, float f21, float f22, float f23,
            float f30, float f31, float f32, float f33) {
        rowsFilled = 0;
        columnsFilled = 0;
        addRow(f00, f01, f02, f03);
        addRow(f10, f11, f12, f13);
        addRow(f20, f21, f22, f23);
        addRow(f30, f31, f32, f33);
    }

    void allZero() {
        for (size_t i = 0; i < 4; i++)
            for (size_t j = 0; j < 4; j++)
                mx[i][j] = 0.0f;
    }

    void addRow(float v0, float v1, float v2, float v3) {
        mx[rowsFilled][0] = v0;
        mx[rowsFilled][1] = v1;
        mx[rowsFilled][2] = v2;
        mx[rowsFilled][3] = v3;
        rowsFilled++;
        columnsFilled = 4;
    }

    void addColumn(float v0, float v1, float v2, float v3) {
        mx[0][columnsFilled] = v0;
        mx[1][columnsFilled] = v1;
        mx[2][columnsFilled] = v2;
        mx[3][columnsFilled] = v3;
        columnsFilled++;
        rowsFilled = 4;
    }

    Matrix4D transposed() const {
        Matrix4D t;
        for (size_t i = 0; i < rowsFilled; i++)
            t.addColumn(mx[i][0], mx[i][1], mx[i][2], mx[i][3]);
        t.rowsFilled = columnsFilled;
        t.columnsFilled = rowsFilled;
        return t;
    }

    Matrix4D operator+(const Matrix4D &m) const {
        Matrix4D sum;
        if (columnsFilled == m.columnsFilled && rowsFilled == m.rowsFilled) {
            sum.rowsFilled = rowsFilled;
            sum.columnsFilled = columnsFilled;
            for (size_t i = 0; i < rowsFilled; i++)
                for (size_t j = 0; j < columnsFilled; j++)
                    sum.mx[i][j] = mx[i][j] + m.mx[i][j];
        }

        return sum;
    }

    Matrix4D operator*(float f) const {
        Matrix4D result;
        result.rowsFilled = rowsFilled;
        result.columnsFilled = columnsFilled;
        for (size_t i = 0; i < result.rowsFilled; i++)
            for (size_t j = 0; j < result.columnsFilled; j++)
                result.mx[i][j] = mx[i][j] * f;
        return result;
    }

    Matrix4D operator*(const Matrix4D &right) const {
        Matrix4D result;
        if (columnsFilled == right.rowsFilled) {
            result.rowsFilled = rowsFilled;
            result.columnsFilled = right.columnsFilled;
            for (size_t i = 0; i < result.rowsFilled; i++)
                for (size_t j = 0; j < result.columnsFilled; j++)
                    for (size_t k = 0; k < columnsFilled; k++)
                        result.mx[i][j] += (mx[i][k] * right.mx[k][j]);
        }
        return result;
    }

    float get(size_t i, size_t j) const {
        return mx[i][j];
    }
};

const int screenWidth = 600;    // alkalmazás ablak felbontása
const int screenHeight = 600;

const size_t maxObjectCount = 10;
const size_t maxLightCount = 3;
const unsigned recursionMax = 6;

const Color worldAmbient = Color(0.2, 0.2, 0.2);
const Color ambientSky = Color(0.3, 0.5, 0.7);

Color image[screenWidth * screenHeight];    // egy alkalmazás ablaknyi kép


struct Ray {
    Vector p0;
    Vector v;
};

class Object;

struct Intersection {
    Vector pos, normal;
    Object const *obj;
    float rayT;
    bool real;
};

class Material {
private:
    Color F0,
            n, k,
            diffuse, ambient, specular;
    float shine;
    bool reflective, refractive;

    void computeF0() {
        Color one = Color(1.0, 1.0, 1.0);
        Color szamlalo = (n - one) * (n - one) + k * k;
        Color nevezo = (n + one) * (n + one) + k * k;
        F0 = szamlalo / nevezo;
    }

    bool isInside(Vector const &n, Vector const &v) {
        float cosAlpha = -(n * v);
        return cosAlpha < NEAR_ZERO;
    }

public:
    Material(Color const &n, Color const &k, Color const &kd, Color const &ka, Color const &ks, float shine, bool isReflective, bool isRefractive)
            : n(n),
              k(k),
              diffuse(kd),
              ambient(ka),
              specular(ks),
              shine(shine),
              reflective(isReflective),
              refractive(isRefractive) {
        computeF0();
    }

    Color reflRadiance(Vector const &l, Vector const &n, Vector const &v, Color const &lIn) const {
        float cosTheta = n * l;
        if (cosTheta < NEAR_ZERO)
            cosTheta = 0.0;
        Color lRef = lIn * diffuse * cosTheta;
        Vector h = (l + v).normalized();
        float cosDelta = n * h;
        if (cosDelta < NEAR_ZERO)
            cosDelta = 0.0;
        lRef = lRef + lIn * specular * powf(cosDelta, shine);
        return lRef;
    }

    Vector reflect(Vector const &n, Vector const &v) const {
        float cosAlpha = -(n * v);
        return (v + n * 2.0 * cosAlpha).normalized();
    }

    Vector refract(Vector const &normal, Vector const &V) {
        Vector N = normal.normalized();
        float cosAlpha = -(N * V);
        float cn = (n.r + n.g + n.b) / 3;
        if (isInside(N, V)) {
            cosAlpha = -cosAlpha;
            N = N * (-1.0f);
            cn = 1.0f / cn;
        }
        float disc = 1 - (1 - cosAlpha * cosAlpha) / cn / cn;
        if (disc < NEAR_ZERO)
            return V.normalized();
        return (V / cn + N * (cosAlpha / cn - sqrtf(disc))).normalized();
    }

    Color Fresnel(Vector const &n, Vector const &v) {
        float cosTheta = (n * v) * (-1.0f);
        Color one = Color(1.0, 1.0, 1.0);
        return F0 + (one - F0) * powf((1 - cosTheta), 5);
    }

    Color const &getAmbient() const {
        return ambient;
    }

    bool isIsReflective() const {
        return reflective;
    }

    bool isIsRefractive() const {
        return refractive;
    }
};

// Forrás: http://www.nicoptere.net/dump/materials.html
const Material gold(Color(0.17, 0.35, 1.5), Color(3.1, 2.7, 1.9),
        Color(0.75, 0.61, 0.23), Color(0.25, 0.20, 0.07), Color(0.63, 0.56, 0.37),
        51.2,
        true, false);

// Forrás: http://www.nicoptere.net/dump/materials.html
const Material silver(Color(0.14, 0.16, 0.13), Color(4.1, 2.3, 3.1),
        Color(0.51, 0.51, 0.51), Color(0.19, 0.19, 0.19), Color(0.51, 0.51, 0.51),
        51.2,
        true, false);

// Forrás: http://globe3d.sourceforge.net/g3d_html/gl-materials__ads.htm
const Material glass(Color(1.5, 1.5, 1.5), Color(0.0, 0.0, 0.0),
        Color(0.59, 0.67, 0.73), Color(0.0, 0.0, 0.0), Color(0.9, 0.9, 0.9),
        96.0,
        true, true);


const Material desk(Color(1.5, 1.5, 1.5), Color(0.0, 0.0, 0.0),
        Color(0.4, 0.4, 0.4), Color(0.2, 0.2, 0.2), Color(0.1, 0.1, 0.1),
        32.0,
        false, false);

const Intersection noIntersection = {
        Vector(0, 0, 0),
        Vector(0, 0, 0),
        NULL,
        -10.0f,
        false
};

float degreeToRad(float degree) {
    return degree * 2.0f * PI / 360.0f;
}

float radToDegree(float rad) {
    return rad * 360.0f / 2.0f / PI;
}

float min(float f1, float f2) {
    return f1 < f2 ? f1 : f2;
}

float max(float f1, float f2) {
    return f1 > f2 ? f1 : f2;
}


class Object {
protected:
    Material material;
public:
    Object() : material((Color()), (Color()), (Color()), (Color()), (Color()), 0, false, false) {
    }

    Object(Material const &material) : material(material) {
    }

    Material const &getMaterial() const {
        return material;
    }

    virtual Intersection intersect(Ray const &ray) = 0;

    virtual Color getTextureModifier(Vector const &at) const {
        return Color(1.0, 1.0, 1.0);
    }
};

class QuadricSurface : public Object {
protected :
    Matrix4D Q;
    Vector limitFrom;
    float limit;
    bool isLimited;

    bool solveQuadraticEquation(float a, float b, float c, float *x1, float *x2) {
        float disc = powf(b, 2.0f) - (4.0f * a * c);
        if (disc < 0.0f || a == 0.0f)
            return false;
        *x1 = (-b - sqrtf(disc)) / (2.0f * a);
        *x2 = (-b + sqrtf(disc)) / (2.0f * a);
        return true;
    }

    float gradX(Vector const &v) {
        Matrix4D dir;
        dir.addRow(1, 0, 0, 0);
        Matrix4D point;
        point.addColumn(v.x, v.y, v.z, 1);

        Matrix4D normalM = (dir * Q * point) + (point.transposed() * Q * dir.transposed());
        return normalM.get(0, 0);
    }

    float gradY(Vector const &v) {
        Matrix4D dir;
        dir.addRow(0, 1, 0, 0);
        Matrix4D point;
        point.addColumn(v.x, v.y, v.z, 1);

        Matrix4D normalM = (dir * Q * point) + (point.transposed() * Q * dir.transposed());
        return normalM.get(0, 0);
    }

    float gradZ(Vector const &v) {
        Matrix4D dir;
        dir.addRow(0, 0, 1, 0);
        Matrix4D point;
        point.addColumn(v.x, v.y, v.z, 1);

        Matrix4D normalM = (dir * Q * point) + (point.transposed() * Q * dir.transposed());
        return normalM.get(0, 0);
    }

    Vector getNormal(Vector const &intersectPoint) {
        float x = gradX(intersectPoint);
        float y = gradY(intersectPoint);
        float z = gradZ(intersectPoint);
        return Vector(x, y, z).normalized();
    }

    Matrix4D stretchTransformInverse(Vector const &stretch) {
        float x = 1.0f / stretch.x;
        float y = 1.0f / stretch.y;
        float z = 1.0f / stretch.z;
        return Matrix4D(
                x, 0, 0, 0,
                0, y, 0, 0,
                0, 0, z, 0,
                0, 0, 0, 1
        );
    }

    Matrix4D offsetTransformInverse(Vector const &offset) {
        float x = -offset.x;
        float y = -offset.y;
        float z = -offset.z;
        return Matrix4D(
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                x, y, z, 1
        );
    }

    Matrix4D rotationTransformInverse(float angle) {
        return Matrix4D(
                cosf(angle), sinf(angle), 0, 0,
                -1.0f * sinf(angle), cosf(angle), 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1
        );
    }

    Matrix4D transformWith(Matrix4D const &Q, Matrix4D const &M) {
        return M * Q * M.transposed();
    }

    void transform(Matrix4D &M, Vector const &stretch, Vector const &offset, float angle) {
        Matrix4D stretchInvM = stretchTransformInverse(stretch);
        Matrix4D offsetInvM = offsetTransformInverse(offset);
        Matrix4D rotationInvM = rotationTransformInverse(angle);
        M = transformWith(M, stretchInvM);
        M = transformWith(M, rotationInvM);
        M = transformWith(M, offsetInvM);
    }

    bool isInsideOrb(Ray const &ray, float t) {
        Vector pos = ray.p0 + (ray.v.normalized() * t);
        float size = (pos - limitFrom).length();
        return size < limit;
    }

    Intersection chooseValidIntersection(float t1, float t2, Ray const &ray) {
        float t = min(t1, t2);
        if (t > NEAR_ZERO && !isLimited) {
            Intersection i;
            i.rayT = t;
            i.real = true;
            i.pos = ray.p0 + (ray.v.normalized() * t);
            i.normal = getNormal(i.pos);
            if (ray.v * i.normal > NEAR_ZERO)
                i.normal *= -1.0f;
            return i;
        }

        if (t > NEAR_ZERO && isLimited && isInsideOrb(ray, t)) {
            Intersection i;
            i.rayT = t;
            i.real = true;
            i.pos = ray.p0 + (ray.v.normalized() * t);
            i.normal = getNormal(i.pos);
            if (ray.v * i.normal > NEAR_ZERO)
                i.normal *= -1.0f;
            return i;
        }

        t = max(t1, t2);
        if (t > NEAR_ZERO && isLimited && isInsideOrb(ray, t)) {
            Intersection i;
            i.rayT = t;
            i.real = true;
            i.pos = ray.p0 + (ray.v.normalized() * t);
            i.normal = getNormal(i.pos);
            if (ray.v * i.normal > NEAR_ZERO)
                i.normal *= -1.0f;
            return i;
        }

        return noIntersection;
    }

public:
    QuadricSurface(Material const &material,
            float a = 0.0f,
            float b = 0.0f,
            float c = 0.0f,
            float d = 0.0f,
            float e = 0.0f,
            float f = 0.0f,
            float g = 0.0f,
            float h = 0.0f,
            float i = 0.0f,
            float j = 0.0f,
            Vector const &stretch = Vector(1.0, 1.0, 1.0),
            Vector const &offset = Vector(0.0, 0.0, 0.0),
            float angle = 0.0f,
            bool isLimited = false,
            Vector const &limitFrom = Vector(0.0, 0.0, 0.0),
            float limit = 1.0
    )
            : Object(material), limitFrom(limitFrom), limit(limit), isLimited(isLimited) {

        Q = Matrix4D(
                a, b, c, d,
                b, e, f, g,
                c, f, h, i,
                d, g, i, j
        );
        transform(Q, stretch, offset, angle);
    }

    Intersection intersect(Ray const &ray) {
        Matrix4D p, v;
        p.addRow(ray.p0.x, ray.p0.y, ray.p0.z, 1.0);
        v.addRow(ray.v.x, ray.v.y, ray.v.z, 0.0);

        Matrix4D aM = v * Q * v.transposed();
        Matrix4D bM = v * Q * p.transposed() * 2.0f;
        Matrix4D cM = p * Q * p.transposed();

        float a = aM.get(0, 0);
        float b = bM.get(0, 0);
        float c = cM.get(0, 0);

        float x1, x2;
        bool valid = solveQuadraticEquation(a, b, c, &x1, &x2);
        if (!valid)
            return noIntersection;

        return chooseValidIntersection(x1, x2, ray);
    }

};

class Sphere : public QuadricSurface {
public:
    Sphere(Material const &material, Vector const &center, float radius)
            : QuadricSurface(material,
            1, 0, 0, 0, 1, 0, 0, 1, 0, -1,
            Vector(radius, radius, radius),
            center
    ) {
    }
};

class Circle : public Object {
    Vector center, normal;
    float radius;

public:

    Circle(Material const &material, Vector const &p0, Vector const &n, float const &radius)
            : Object(material), center(p0), normal(n.normalized()), radius(radius) {
    }

    Intersection intersect(Ray const &ray) {
        Intersection i;
        float disc = normal * (ray.v.normalized());
        if (disc == 0.0)
            return noIntersection;
        float t = -1.0f * (normal * (ray.p0 - center)) * (1.0f / disc);
        if (t > NEAR_ZERO) {
            Vector intersectPos = ray.p0 + (ray.v.normalized() * t);
            if ((intersectPos - center).length() <= radius) {
                i.real = true;
                i.normal = normal;
                i.rayT = t;
                i.pos = intersectPos;
                return i;
            }
        }
        return noIntersection;
    }

    Color getTextureModifier(Vector const &at) const {
        float dist = (at - center).length();
        float stripeWidth = 0.2; //radius / 800.0f;
        int stripeNr = (int) floorf(dist / stripeWidth);
        if (stripeNr % 2 == 0)
            return Color(1.0, 1.0, 1.0);
        return Color(6.0, 6.0, 6.0);
    }
};

class Cylinder : public QuadricSurface {
public:

    Cylinder(Material const &material = glass)
            : QuadricSurface(material) {
    }

    Cylinder(Material const &material, Vector const &center, float radius, float height, float clockWiseAngle)
            : QuadricSurface(
            material,
            1, 0, 0, 0, 0, 0, 0, 1, 0, -1,
            Vector(radius, 1.0, radius),
            center,
            clockWiseAngle,
            true) {
        float counterClockWiseAngle = 2 * PI - (clockWiseAngle - degreeToRad(90.0f));
        Vector direction = Vector(cosf(counterClockWiseAngle), sinf(counterClockWiseAngle), 0.0f);
        direction = direction.normalized();
        Vector top = center + (direction * height);
        Vector bottom = center;
        Vector middle = (top + bottom) / 2.0f;
        limitFrom = middle;
        float a = radius;
        float b = (top - middle).length();
        float c = sqrtf(a * a + b * b);
        limit = c;
    }
};

class Ellipsoid : public QuadricSurface {
    Vector center;
public:
    Ellipsoid(Material const &material,
            Vector const &center,
            Vector const &size,
            float angle)
            : QuadricSurface(
            material,
            1, 0, 0, 0, 1, 0, 0, 1, 0, -1,
            size,
            center,
            angle
    ), center(center) {
    }

    Intersection findSurfacePointNear(Vector near) {
        Ray r;
        r.p0 = near;
        r.v = (center - near).normalized();
        return intersect(r);
    }
};

class Paraboloid : public QuadricSurface {
    Vector center;
    float height;
    float clockWiseAngle;
public:
    Paraboloid(Material const &material = glass)
            : QuadricSurface(material) {
    }

    Paraboloid(Material const &material, Vector const &center, float radius, float height, float clockWiseAngle)
            : QuadricSurface(
            material,
            1, 0, 0, 0, 0, 0, -0.5f, 1, 0, 0,
            Vector(radius / sqrtf(height), 1.0f, radius / sqrtf(height)),
            center,
            clockWiseAngle,
            true),
              center(center), height(height), clockWiseAngle(clockWiseAngle) {
        limitFrom = center;
        float a = radius;
        float b = height;
        float c = sqrtf(a * a + b * b);
        limit = c;
    }

    Intersection findSurfacePointNear(Vector near) {
        Ray r;
        r.p0 = near;
        float counterClockWiseAngle = 2 * PI - (clockWiseAngle - degreeToRad(90.0f));
        Vector direction = Vector(cosf(counterClockWiseAngle), sinf(counterClockWiseAngle), 0.0f).normalized();
        Vector middle = center + (direction * height / 2.0f);
        r.v = (middle - near).normalized();
        return intersect(r);
    }
};

struct CylinderCactus {
    Object *objects[3];
    size_t objectSize;

    CylinderCactus() {
        objectSize = 0;
    }

    void setPos(Vector bottomCenter, float radius, float height) {
        objects[objectSize++] = new Cylinder(glass, bottomCenter, radius, height, degreeToRad(0.0f));
        //objects[objectSize++] = new Circle(gold, bottomCenter, Vector(0,1,0), radius);
        //objects[objectSize++] = new Circle(glass, bottomCenter + Vector(0, height, 0), Vector(0,1,0), radius);

        bottomCenter = Vector(bottomCenter.x + radius, bottomCenter.y + height * 0.6f, bottomCenter.z);
        radius *= 0.45f;
        height *= 0.45f;
        objects[objectSize++] = new Cylinder(glass, bottomCenter, radius, height, degreeToRad(90.0f));

        bottomCenter = Vector(bottomCenter.x + height * 0.7f, bottomCenter.y + radius, bottomCenter.z);
        radius *= 0.7f;
        height *= 0.45f;
        objects[objectSize++] = new Cylinder(glass, bottomCenter, radius, height, degreeToRad(0.0f));
    }

    ~CylinderCactus() {
        for (size_t i = 0; i < objectSize; i++)
            delete objects[i];
    }
} cylinderCactus;

struct EllipsoidCactus {
    Object *objects[3];
    size_t objectSize;

    EllipsoidCactus() {
        objectSize = 0;
    }

    void setPos(Vector bottomPoint, float height) {
        Vector middle = bottomPoint + Vector(0, height / 2.0f, 0);
        Vector size = Vector(height / 4.0f, height / 2.0f, height / 4.0f);
        Ellipsoid *e = new Ellipsoid(gold, middle, size, degreeToRad(0.0f));
        objects[objectSize++] = e;

        Vector meroleges = Vector(1, 0, 0);
        Vector mostRightPoint = middle + (meroleges * height / 4.0f);
        Vector pointNearSurface = mostRightPoint + Vector(0.1f, height * 0.4f, 0.0f);
        Intersection i = e->findSurfacePointNear(pointNearSurface);
        Vector normal = i.normal.normalized();
        float angle = acosf(normal * Vector(0, 1, 0));
        bottomPoint = i.pos;
        height *= 0.55;
        middle = bottomPoint + (normal * height / 2.0f);
        size = Vector(height / 4.0f, height / 2.0f, height / 4.0f);
        e = new Ellipsoid(gold, middle, size, angle);
        objects[objectSize++] = e;

        meroleges = Vector(0, 0, 1) % normal;
        Vector mostTopPoint = middle + (meroleges * height / 4.0f);
        pointNearSurface = mostTopPoint + (normal * height * 0.3);
        i = e->findSurfacePointNear(pointNearSurface);
        normal = i.normal.normalized();
        angle = acosf(normal * Vector(0, 1, 0));
        bottomPoint = i.pos;
        height *= 0.4;
        middle = bottomPoint + (normal * height / 2.0f);
        size = Vector(height / 4.0f, height / 2.0f, height / 4.0f);
        e = new Ellipsoid(gold, middle, size, angle);
        objects[objectSize++] = e;
    }

    ~EllipsoidCactus() {
        for (size_t i = 0; i < objectSize; i++)
            delete objects[i];
    }
} ellipsoidCactus;

struct ParaboloidCactus {
    Object *objects[3];
    size_t objectSize;

    ParaboloidCactus() {
        objectSize = 0;
    }

    void setPos(Vector bottomPoint, float radiusOnTop, float height) {
        Paraboloid *p = new Paraboloid(silver, bottomPoint, radiusOnTop, height, degreeToRad(0.0f));
        objects[objectSize++] = p;
    }

    ~ParaboloidCactus() {
        for (size_t i = 0; i < objectSize; i++)
            delete objects[i];
    }
} paraboloidCactus;

class Camera {
    Vector eye, lookat, up, right;
    float width, height;

public:
    Camera() {
    }

private:
    Vector getPosOnScreen(unsigned X, unsigned Y) {
        // Az ernyő melyik pontja felel meg egy pixelnek?
        float screenPosX = ((float) X + 0.5f - screenWidth / 2.0f)
                / (screenWidth / 2.0f);
        float screenPosY = ((float) Y + 0.5f - screenHeight / 2.0f)
                / (screenHeight / 2.0f);

        // Az ernyő is a világ része, mi a világkoordinátája a pontnak?
        return lookat + (right * screenPosX) + (up * screenPosY);
    }

public:
    Camera(Vector const &eye, Vector const &lookat, Vector const &up, Vector const &right, float width, float height)
            : eye(eye), lookat(lookat), up(up), right(right), width(width), height(height) {
    }

    Ray getRay(unsigned X, unsigned Y) {
        Ray r;
        Vector posOnScreen = getPosOnScreen(X, Y);
        r.p0 = eye;
        r.v = (posOnScreen - eye).normalized();
        return r;
    }
};

class Light {
    Color color;
    Vector p;

public:

    Light() {
    }

    Light(Color const &color, Vector const &p) : color(color), p(p) {
    }

    Color getRad(Vector const &x) const {
        float distance = (x - p).length();
        return color / powf(distance, 2) / 10;
    }

    Vector const &getP() const {
        return p;
    }
};

class Scene {
    Object *objects[maxObjectCount];
    Light *lights[maxLightCount];
    size_t objectSize, lightSize;
    Camera *camera;

    Color directIllumination(Ray const &ray, Intersection const &hit) const {
        Color color = hit.obj->getMaterial().getAmbient() * worldAmbient;
        Vector x = hit.pos;
        Vector N = hit.normal.normalized();

        for (size_t i = 0; i < lightSize; i++) {
            Ray shadowRay;
            shadowRay.p0 = x;
            shadowRay.v = (lights[i]->getP() - x).normalized();
            Intersection shadowHit = intersectAll(shadowRay);
            Vector y = shadowHit.pos;
            if (!shadowHit.real ||
                    ((x - y).length() > (x - lights[i]->getP()).length())) {
                Vector V = ray.v.normalized() * (-1.0f);
                Vector L = shadowRay.v.normalized();
                color += hit.obj->getMaterial().reflRadiance(L, N, V, lights[i]->getRad(x));
            }
        }
        return color;
    }

    Color reflectColor(Intersection const &hit, Ray const &ray, int d) const {
        Color color(0, 0, 0);
        Material material = hit.obj->getMaterial();
        if (material.isIsReflective()) {
            Ray reflectedRay;
            reflectedRay.v = material.reflect(hit.normal, ray.v);
            reflectedRay.p0 = hit.pos;
            Color Fresnel = material.Fresnel(hit.normal, ray.v);
            color += Fresnel * trace(reflectedRay, d + 1);
        }
        return color;
    }

    Color refractColor(Intersection const &hit, Ray const &ray, int d) const {
        Color color(0, 0, 0);
        Material material = hit.obj->getMaterial();

        if (material.isIsRefractive()) {
            Ray refractedRay;
            refractedRay.v = material.refract(hit.normal, ray.v);
            refractedRay.p0 = hit.pos;
            Color Fresnel = material.Fresnel(hit.normal, ray.v);
            Color one = Color(1, 1, 1);
            color += (one - Fresnel) * trace(refractedRay, d + 1);
        }
        return color;
    }

    Color trace(Ray const &ray, int d) const {
        if (d > recursionMax)
            return worldAmbient;

        Intersection hit = intersectAll(ray);
        if (!hit.real)
            return ambientSky;

        Color color = directIllumination(ray, hit);
        color += reflectColor(hit, ray, d);
        color += refractColor(hit, ray, d);
        color *= hit.obj->getTextureModifier(hit.pos);
        return color;
    }

    Intersection intersectAll(Ray const &ray) const {
        Intersection closest;
        closest.rayT = FLT_MAX;
        closest.real = false;
        for (size_t i = 0; i < objectSize; i++) {
            Intersection inters = objects[i]->intersect(ray);
            if (inters.real && inters.rayT < closest.rayT) {
                closest = inters;
                closest.obj = objects[i];
            }
        }
        return closest;
    }

    void add(Object *object) {
        objects[objectSize++] = object;
    }

    void add(Light *light) {
        lights[lightSize++] = light;
    }

public:

    Scene() {
        objectSize = lightSize = 0;
    }

    void render() {
        for (size_t Y = 0; Y < screenHeight; Y++)
            for (size_t X = 0; X < screenWidth; X++) {
                Ray ray = camera->getRay(X, Y);
                Color color = trace(ray, 0);
                //color = color/(Color(1,1,1) + color);
                // TODO: tone Mapping
                image[Y * screenWidth + X] = color;
            }
    }

    void build() {
        Vector cylinderPos(-1.4, -0.9, 1.5);
        Vector ellipsoidPos(0.0, -0.9, 2.0);
        Vector paraboloidPos(1.0, -0.9, 1.0);

        // TODO paraboloid, fények
        // asztal
        add(new Circle(desk, Vector(0.0, -1.0, 0.0), Vector(0.0, 1.0, 0.0), 6.0));

        // henger-kaktusz
        cylinderCactus.setPos(cylinderPos, 0.5f, 2.0f);
        for (size_t i = 0; i < cylinderCactus.objectSize; i++)
            add(cylinderCactus.objects[i]);

        // ellipszoid-kaktusz
        ellipsoidCactus.setPos(ellipsoidPos, 2.0f);
        for (size_t i = 0; i < ellipsoidCactus.objectSize; i++)
            add(ellipsoidCactus.objects[i]);

        paraboloidCactus.setPos(paraboloidPos, 0.5f, 2.0f);
        add(paraboloidCactus.objects[0]);


        // fényforrások
        // henger mögött
        //add(new Light(Color(17, 17, 26), cylinderPos + Vector(-0.5, 2.9, 1.0)));
        //henger-lámpa
        //add(new Light(Color(2,2,5), cylinderPos + Vector(0, 1.7, 0)));
        // hengerek találkozásánál
        add(new Light(Color(2, 2, 5), cylinderPos + Vector(0.48, 1.0, 0.0)));
        // hiba: teljes visszaverődés?
        //add(new Light(Color(2,2,5), cylinderPos + Vector(0.50, 1.0, 0.0)));

        // ellipszoid mögött:
        add(new Light(Color(12, 4, 4), ellipsoidPos + Vector(1.0, 3.0, 1.0)));

        // kamera
        // középen
        /*
         Vector lookat = Vector(0, 0, 0);
         Vector eye = Vector(0, 0, -2);
         Vector right = Vector(1, 0, 0);
         Vector dir = (lookat - eye).normalized();
         Vector up = (dir % right).normalized();
         */


        // döntve
        Vector lookat = Vector(0, 0.9, 0);
        Vector eye = Vector(0, 1.5, -0.7);
        Vector right = Vector(1, 0, 0);
        Vector dir = (lookat - eye).normalized();
        Vector up = (dir % right).normalized();
        //right *= 1.5;
        //up *= 1.5;

        camera = new Camera(eye, lookat, up, right, screenWidth, screenHeight);
    }
} scene;

// Inicializacio, a program futasanak kezdeten, az OpenGL kontextus letrehozasa utan hivodik meg (ld. main() fv.)
void onInitialization() {
    glViewport(0, 0, screenWidth, screenHeight);
    scene.build();
    scene.render();
}

// Rajzolas, ha az alkalmazas ablak ervenytelenne valik, akkor ez a fuggveny hivodik meg
void onDisplay() {
    glDrawPixels(screenWidth, screenHeight, GL_RGB, GL_FLOAT, image);
    glutSwapBuffers();                    // Buffercsere: rajzolas vege

}

// Billentyuzet esemenyeket lekezelo fuggveny (lenyomas)
void onKeyboard(unsigned char key, int x, int y) {
    if (key == 'd') glutPostRedisplay();        // d beture rajzold ujra a kepet

}

// Billentyuzet esemenyeket lekezelo fuggveny (felengedes)
void onKeyboardUp(unsigned char key, int x, int y) {

}

// Eger esemenyeket lekezelo fuggveny
void onMouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN)   // A GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON illetve GLUT_DOWN / GLUT_UP
        glutPostRedisplay();                         // Ilyenkor rajzold ujra a kepet
}

// Eger mozgast lekezelo fuggveny
void onMouseMotion(int x, int y) {

}

// `Idle' esemenykezelo, jelzi, hogy az ido telik, az Idle esemenyek frekvenciajara csak a 0 a garantalt minimalis ertek
void onIdle() {
}

// ...Idaig modosithatod
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A C++ program belepesi pontja, a main fuggvenyt mar nem szabad bantani
int main(int argc, char **argv) {
    glutInit(&argc, argv);                // GLUT inicializalasa
    glutInitWindowSize(600, 600);            // Alkalmazas ablak kezdeti merete 600x600 pixel
    glutInitWindowPosition(100, 100);            // Az elozo alkalmazas ablakhoz kepest hol tunik fel
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);    // 8 bites R,G,B,A + dupla buffer + melyseg buffer

    glutCreateWindow("Grafika hazi feladat");        // Alkalmazas ablak megszuletik es megjelenik a kepernyon

    glMatrixMode(GL_MODELVIEW);                // A MODELVIEW transzformaciot egysegmatrixra inicializaljuk
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);            // A PROJECTION transzformaciot egysegmatrixra inicializaljuk
    glLoadIdentity();

    onInitialization();                    // Az altalad irt inicializalast lefuttatjuk


    glutDisplayFunc(onDisplay);                // Esemenykezelok regisztralasa
    glutMouseFunc(onMouse);
    //glutIdleFunc(onIdle);
    glutKeyboardFunc(onKeyboard);
    glutKeyboardUpFunc(onKeyboardUp);
    glutMotionFunc(onMouseMotion);

    glutMainLoop();                    // Esemenykezelo hurok


    return 0;
}
