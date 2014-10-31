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

    Vector operator*(float a) {
        return Vector(x * a, y * a, z * a);
    }

    Vector operator/(float a) {
        return Vector(x / a, y / a, z / a);
    }

    Vector operator+(const Vector &v) {
        return Vector(x + v.x, y + v.y, z + v.z);
    }

    Vector operator-(const Vector &v) {
        return Vector(x - v.x, y - v.y, z - v.z);
    }

    float operator*(const Vector &v) {    // dot product
        return (x * v.x + y * v.y + z * v.z);
    }

    Vector operator%(const Vector &v) {    // cross product
        return Vector(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
    }

    float length() {
        return sqrt(x * x + y * y + z * z);
    }

    Vector normalized() {
        return *this / length();
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

    Color operator*(float a) {
        return Color(r * a, g * a, b * a);
    }

    Color operator/(float a) {
        return Color(r / a, g / a, b / a);
    }

    Color operator*(const Color &c) {
        return Color(r * c.r, g * c.g, b * c.b);
    }

    Color operator/(const Color &c) {
        return Color(r / c.r, g / c.g, b / c.b);
    }

    Color operator+(const Color &c) {
        return Color(r + c.r, g + c.g, b + c.b);
    }

    Color operator-(const Color &c) {
        return Color(r - c.r, g - c.g, b - c.b);
    }
};

const int screenWidth = 600;    // alkalmazás ablak felbontása
const int screenHeight = 600;

const size_t maxObjectCount = 10;
const size_t maxLightCount = 3;

Color image[screenWidth * screenHeight];    // egy alkalmazás ablaknyi kép

struct Ray {
    Vector p0;
    Vector v;
};

struct Intersection {
    Vector pos, normal;
    Object *obj;
    float rayT;
};

class Material {
private:
    Color F0, n, k, ka, kd, ks;
    float shine;
    bool isReflective, isRefractive;

    void computeF0() {
        Color one = Color(1.0, 1.0, 1.0);
        Color szamlalo = (n - one) * (n - one) + k * k;
        Color nevezo   = (n + one) * (n + one) + k * k;
        F0 = szamlalo / nevezo;
    }
public:
    Material() {
    }

    Material(Color const &F0, Color const &n, Color const &kd, Color const &ks, float shine, bool isReflective, bool isRefractive)
            : F0(F0), n(n), kd(kd), ks(ks), shine(shine), isReflective(isReflective), isRefractive(isRefractive) {
    }

    Color reflRadiance(Vector *l, Vector *n, Vector *v, Color *lIn) {
        // TODO
    }

    Vector reflect(Vector *n, Vector *v) {
        // TODO
    }

    Vector refract(Vector *n, Vector *v) {
        // TODO
    }

    Color Fresnel(Vector *n, Vector *v) {
        // TODO
    }


    Color const &getF0() const {
        return F0;
    }

    void setF0(Color const &F0) {
        Material::F0 = F0;
    }

    Color const &getN() const {
        return n;
    }

    void setN(Color const &n) {
        Material::n = n;
    }

    Color const &getKd() const {
        return kd;
    }

    void setKd(Color const &kd) {
        Material::kd = kd;
    }

    Color const &getKs() const {
        return ks;
    }

    void setKs(Color const &ks) {
        Material::ks = ks;
    }

    float getShine() const {
        return shine;
    }

    void setShine(float shine) {
        Material::shine = shine;
    }

    bool isIsReflective() const {
        return isReflective;
    }

    void setIsReflective(bool isReflective) {
        Material::isReflective = isReflective;
    }

    bool isIsRefractive() const {
        return isRefractive;
    }

    void setIsRefractive(bool isRefractive) {
        Material::isRefractive = isRefractive;
    }
};

class Object {
protected:
    Material material;
public:
    Object() {
    }

    Object(Material const &material) : material(material) {
    }

    Material const &getMaterial() const {
        return material;
    }

    void setMaterial(Material const &material) {
        Object::material = material;
    }

    virtual Vector intersect(Ray *ray, Vector *normal) {
    };

};

class Plane : Object {
    Vector p, n;

public:
    Plane(Material const &material, Vector const &p, Vector const &n) : Object(material), p(p), n(n) {
    }

    Vector intersect(Ray *ray, Vector *normal) {
        // TODO
    }

    Vector const &getP() const {
        return p;
    }

    void setP(Vector const &p) {
        Plane::p = p;
    }

    Vector const &getN() const {
        return n;
    }

    void setN(Vector const &n) {
        Plane::n = n;
    }
};

class Cylinder : Object {
    Vector center, direction;
    float radius, height;

public:
    Cylinder(Material const &material, Vector const &center, float radius, float height, Vector const &direction)
            : Object(material), center(center), radius(radius), height(height), direction(direction) {
    }

    Vector intersect(Ray *ray, Vector *normal) {
        // TODO
    }

    Vector const &getCenter() const {
        return center;
    }

    void setCenter(Vector const &center) {
        Cylinder::center = center;
    }

    float getRadius() const {
        return radius;
    }

    void setRadius(float radius) {
        Cylinder::radius = radius;
    }

    float getHeight() const {
        return height;
    }

    void setHeight(float height) {
        Cylinder::height = height;
    }

    Vector const &getDirection() const {
        return direction;
    }

    void setDirection(Vector const &direction) {
        Cylinder::direction = direction;
    }

};

class Ellipsoid : Object {
    Vector focus1;
    Vector focus2;
    float distance;


public:
    Ellipsoid(Material const &material, Vector const &focus1, Vector const &focus2, float distance)
            : Object(material), focus1(focus1), focus2(focus2), distance(distance) {
    }

    Vector intersect(Ray *ray, Vector *normal) {
        // TODO
    }

    Vector const &getFocus1() const {
        return focus1;
    }

    void setFocus1(Vector const &focus1) {
        Ellipsoid::focus1 = focus1;
    }

    Vector const &getFocus2() const {
        return focus2;
    }

    void setFocus2(Vector const &focus2) {
        Ellipsoid::focus2 = focus2;
    }

    float getDistance() const {
        return distance;
    }

    void setDistance(float distance) {
        Ellipsoid::distance = distance;
    }
};

class Paraboloid : Object {
    Vector plane, focus;
    float height;

public:
    Paraboloid(Material const &material, Vector const &plane, Vector const &focus, float height)
            : Object(material), plane(plane), focus(focus), height(height) {
    }

    Vector intersect(Ray *ray, Vector *normal) {
        // TODO
    }

    Vector const &getPlane() const {
        return plane;
    }

    void setPlane(Vector const &plane) {
        Paraboloid::plane = plane;
    }

    Vector const &getFocus() const {
        return focus;
    }

    void setFocus(Vector const &focus) {
        Paraboloid::focus = focus;
    }

    float getHeight() const {
        return height;
    }

    void setHeight(float height) {
        Paraboloid::height = height;
    }
};

class Camera {
    Vector eye, lookat, up, right;
    float width, height;

    Vector getPosOnScreen(unsigned X, unsigned Y) {
        // Az ernyő melyik pontja felel meg egy pixelnek?
        float screenPosX = (float) (((float)X + 0.5 - screenWidth / 2.0)
                / ((float) screenWidth / 2.0));
        float screenPosY = (float) (((float)X + 0.5 - screenWidth / 2.0)
                / ((float) screenWidth / 2.0));

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
        Vector direction = (posOnScreen - eye).normalized();
        r.p0 = posOnScreen;
        r.v = direction;
        return r;
    }

    Vector const &getEye() const {
        return eye;
    }

    void setEye(Vector const &eye) {
        Camera::eye = eye;
    }

    Vector const &getLookat() const {
        return lookat;
    }

    void setLookat(Vector const &lookat) {
        Camera::lookat = lookat;
    }

    Vector const &getUp() const {
        return up;
    }

    void setUp(Vector const &up) {
        Camera::up = up;
    }

    Vector const &getRight() const {
        return right;
    }

    void setRight(Vector const &right) {
        Camera::right = right;
    }

    float getWidth() const {
        return width;
    }

    void setWidth(float width) {
        Camera::width = width;
    }

    float getHeight() const {
        return height;
    }

    void setHeight(float height) {
        Camera::height = height;
    }
};

class Light {
    Color color;
    Vector p;

// var?: lOut, type
public:

    Light() {
    }

    Light(Vector const &p) : p(p) {
    }

    Vector getDir() {
        //TODO
    }

    Color getRad(Vector x) {
        //TODO
    }

    Color const &getColor() const {
        return color;
    }

    void setColor(Color const &color) {
        Light::color = color;
    }

    Vector const &getP() const {
        return p;
    }

    void setP(Vector const &p) {
        Light::p = p;
    }
};

class Scene {
    Object *objects[maxObjectCount];
    Light lights[maxLightCount];
    size_t objectSize, lightSize;
    Camera camera;

    Color trace(Ray const &ray, int d) {
        // TODO
    }

public:
    Scene(Camera const &camera) : camera(camera) {
        objectSize = lightSize = 0;
    }

    void add(Object *object) {
        objects[objectSize++] = object;
    }

    void add(Light const &light) {
        lights[lightSize++] = light;
    }

    void render() {
        for (size_t Y = 0; Y < screenHeight; Y++)
            for (size_t X = 0; X < screenWidth; X++) {
                Ray ray = camera.getRay(X, Y);
                image[Y * screenWidth + X] = trace(ray, 0);
            }
    }

    // TODO intersectAll

    void build() {
        // TODO
    }
};

// Inicializacio, a program futasanak kezdeten, az OpenGL kontextus letrehozasa utan hivodik meg (ld. main() fv.)
void onInitialization() {
    glViewport(0, 0, screenWidth, screenHeight);

    // Peldakent keszitunk egy kepet az operativ memoriaba
    for (int Y = 0; Y < screenHeight; Y++)
        for (int X = 0; X < screenWidth; X++)
            image[Y * screenWidth + X] = Color((float) X / screenWidth, (float) Y / screenHeight, 0);

}

// Rajzolas, ha az alkalmazas ablak ervenytelenne valik, akkor ez a fuggveny hivodik meg
void onDisplay() {
    glClearColor(0.1f, 0.2f, 0.3f, 1.0f);        // torlesi szin beallitasa
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // kepernyo torles

    // ..

    // Peldakent atmasoljuk a kepet a rasztertarba
    glDrawPixels(screenWidth, screenHeight, GL_RGB, GL_FLOAT, image);
    // Majd rajzolunk egy kek haromszoget
    glColor3f(0, 0, 1);
    glBegin(GL_TRIANGLES);
    glVertex2f(-0.2f, -0.2f);
    glVertex2f(0.2f, -0.2f);
    glVertex2f(0.0f, 0.2f);
    glEnd();

    // ...

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
    long time = glutGet(GLUT_ELAPSED_TIME);        // program inditasa ota eltelt ido

}

// ...Idaig modosithatod
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// A C++ program belepesi pontja, a main fuggvenyt mar nem szabad bantani
int main(int argc, char **argv) {
    glutInit(&argc, argv); 				// GLUT inicializalasa
    glutInitWindowSize(600, 600);			// Alkalmazas ablak kezdeti merete 600x600 pixel 
    glutInitWindowPosition(100, 100);			// Az elozo alkalmazas ablakhoz kepest hol tunik fel
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);	// 8 bites R,G,B,A + dupla buffer + melyseg buffer

    glutCreateWindow("Grafika hazi feladat");		// Alkalmazas ablak megszuletik es megjelenik a kepernyon

    glMatrixMode(GL_MODELVIEW);				// A MODELVIEW transzformaciot egysegmatrixra inicializaljuk
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);			// A PROJECTION transzformaciot egysegmatrixra inicializaljuk
    glLoadIdentity();

    onInitialization();					// Az altalad irt inicializalast lefuttatjuk

    glutDisplayFunc(onDisplay);				// Esemenykezelok regisztralasa
    glutMouseFunc(onMouse); 
    glutIdleFunc(onIdle);
    glutKeyboardFunc(onKeyboard);
    glutKeyboardUpFunc(onKeyboardUp);
    glutMotionFunc(onMouseMotion);

    glutMainLoop();					// Esemenykezelo hurok
    
    return 0;
}

