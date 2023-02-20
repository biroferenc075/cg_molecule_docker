//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Biró Ferenc
// Neptun : HR4VCG
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
#include "framework.h"

// vertex shader in GLSL: It is a Raw string (C++11) since it contains new line characters
const char * const vertexSourceWorld = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
    layout(location = 1) in vec3 vc;
    out vec3 vColor;
	void main() {
        vec4 e = vec4(vp.x, vp.y, 0, 1) * MVP;
        float w = sqrt(e.x*e.x+e.y*e.y)+1;
        vec4 h = vec4(e.x, e.y, 0, 1);

		gl_Position = vec4(h.x/(w+1), h.y/(w+1), 0, 1);		// transform vp from modeling space to normalized device space
        //gl_Position = vec4(vp.x, vp.y, 0,1) * MVP;
        vColor = vc;
}
)";

const char * const vertexSourceCircle = R"(
	#version 330				// Shader 3.3
	precision highp float;		// normal floats, makes no difference on desktop computers

	uniform mat4 MVP;			// uniform variable, the Model-View-Projection transformation matrix
	layout(location = 0) in vec2 vp;	// Varying input: vp = vertex position is expected in attrib array 0
    layout(location = 1) in vec3 vc;
    out vec3 vColor;
	void main() {
        gl_Position = vec4(vp.x, vp.y, 0, 1);
        vColor = vc;
}
)";

// fragment shader in GLSL
const char * const fragmentSource = R"(
	#version 330			// Shader 3.3
	precision highp float;	// normal floats, makes no difference on desktop computers
	
	uniform vec3 color;		// uniform variable, the color of the primitive
	out vec4 outColor;		// computed color of the current pixel
    in vec3 vColor;
	void main() {
		outColor = vec4(vColor, 1);	// computed color is the color of the primitive
	}
)";

GPUProgram worldProgram; // vertex and fragment shaders
GPUProgram circleProgram;
unsigned int vao;	   // virtual world on the GPU

const float mH = 1.67E-27; // kg
const float cE = 1.6E-19;  // c
const float k = 9E9 ;      // Nm^2/c^2
const float vW = 1.0E-3;     // N*s/m
const float distC = 1E-9;   // m
const float cFC = -0.0138;//-3.45; //
const float airResC = 0.05f;
const float tC = 1E-2;  //s
const int maxMass = 10;
const int maxCharge = 10;
const int maxDistance = 10;
const int lRes = 8;
const int cRes = 24;
const int pRes = 128;
const vec3 lineColor = vec3(1.0, 1.0, 1.0);
const vec3 circleColor = vec3(0.5, 0.5,0.5);
const vec3 bgrColor = vec3(0.0,0.0,0.0);
const float atomRad = 0.1;
const float maxRad = 1.0f;
float xOffs = 0.0;
float yOffs = 0.0;
bool drawLines = true;
bool drawCircles = true;

typedef struct Vertex {
    vec2 pos;
    vec3 col;
    Vertex(vec2 p, vec3 c) {
        pos = p;
        col = c;
    }
    Vertex() {
        pos = (0,0);
        col = (0, 0, 0);
    }
    Vertex(vec2 p) {
        pos = p;
        col = (0, 0, 0);
    }
    void move(vec2 d) {
        //printf("%f %f -> %f %f\n",pos.x, pos.y, vec2(pos+d).x, vec2(pos+d).y);
        pos=pos+d;
    }
    void rot(float a, vec2 o) {
        float c = cosf(a);
        float s = sinf(a);
        vec2 d = pos-o;
        vec4 rot = vec4(d.x, d.y,0,0)*mat4(c,s,0,0,-s,c,0,0,0,0,1,0,0,0,0,1);
        pos = vec2(rot.x, rot.y)+o;
        //pos = ();
    }
};

typedef struct Atom {
    int mass;
    int charge;
    Atom* parent;
    std::vector<Atom*> children = std::vector<Atom*>();
    Vertex vertex = Vertex();

    void move(vec2 d) {
        //printf("%f\n", d.x);
        vertex.move(d);
        for(int i = 0; i < children.size(); i++)
            children[i]->move(d);
    }
    void rot(float a, vec2 o) {
        vertex.rot(a, o);
        for(int i = 0; i < children.size(); i++)
            children[i]->rot(a, o);
    }
    vec2 relPos(Atom a) {
        return a.vertex.pos-vertex.pos;
    }
    float calcMOI(float in, vec2 axis) {
        float res = in;
        res+=pow(length(vertex.pos-axis),2)*mass;
        for(int i = 0; i < children.size(); i++)
            res=children[i]->calcMOI(res, axis);
        return res;
    }
} Atom;


typedef struct Molecule {
    Atom* root;
    int mass;
    float MOI = 0;
    vec2 COM = vec2(0,0);
    std::vector<Atom*> atoms = std::vector<Atom*>();
    std::vector<Vertex> edgeVertices = std::vector<Vertex>();
    std::vector<Vertex> cirVertices = std::vector<Vertex>();
    std::vector<Vertex> vertices= std::vector<Vertex>();
    void move(vec2 d) {
        root->move(d);
        COM=COM+d;
        for(int i = 0; i < edgeVertices.size(); i++)
            edgeVertices[i].move(d);
        for(int i = 0; i < cirVertices.size(); i++)
            cirVertices[i].move(d);
        for(int i = 0; i < vertices.size(); i++)
            vertices[i].move(d);
    }
    vec2 vel = (0,0);
    void rot(float a) {
        root->rot(a, COM);
        for(int i = 0; i < edgeVertices.size(); i++)
            edgeVertices[i].rot(a, COM);
        for(int i = 0; i < cirVertices.size(); i++)
            cirVertices[i].rot(a, COM);
        for(int i = 0; i < vertices.size(); i++)
            vertices[i].rot(a, COM);
    }
    vec2 relPos(vec2 p) {
        return p-COM;
    }
    vec2 relPos(Atom a) {
        return a.vertex.pos-COM;
    }
    float aVel = 0;
    void calcMOI() {
        MOI=root->calcMOI(0,COM);
        //printf("%f \n", MOI);
    }
} Molecule;

std::vector<Vertex> verticesVec = std::vector<Vertex>();
std::vector<Vertex> plate = std::vector<Vertex>();
Vertex* vertices;
std::vector<Molecule*> mols = std::vector<Molecule*>();
Molecule* mol;
//https://hu.wikipedia.org/wiki/Pr%C3%BCfer-k%C3%B3d Prüfer-kódból fa 1. módszer pszeudókód alapján
void generateMolecule(Molecule& res) {
    try {
        //printf("begin");
        int atomNum = rand() % 6 + 2;
        //printf("#%d\n", atomNum);
        int sumCharge = 0;
        int sumMass = 0;
        Atom **atoms = new Atom *[atomNum];
        for (int i = 0; i < atomNum; i++) {
            atoms[i] = new Atom;
            float x = static_cast <float> (2 * maxRad * rand()) / static_cast <float> (RAND_MAX) - maxRad;
            float y = static_cast <float> (2 * maxRad * rand()) / static_cast <float> (RAND_MAX) - maxRad;
            int mass = rand() % (maxMass-1)+1;
            sumMass += mass;
            int ch = rand() % maxCharge * 2 - maxCharge;
            if(i == atomNum-1) {
                ch = sumCharge*-1;
                res.mass = sumMass;
            }
            sumCharge += ch;
            atoms[i]->charge = ch;
            atoms[i]->mass = mass;
            vec3 dColor = ch > 0 ? vec3(0.8*(float)ch/maxCharge, 0.0, 0.0) : vec3(0.0, 0.0,-0.8*(float)(ch)/maxCharge);
            vec3 bColor = vec3(0.2,0.2,0.2);
            Vertex v = Vertex(vec2(x, y), bColor+dColor);
            atoms[i]->vertex = v;
            res.vertices.push_back(v);
            res.COM = res.COM+vec2(x,y)/res.vertices.size();
            res.atoms.push_back(atoms[i]);
        }
        if(atomNum > 3) {
            //printf("hey!\n");
            int indices[2 * (atomNum - 2)+1];
            for (int i = 0; i < atomNum - 2; i++) {
                indices[i] = rand() % atomNum;
            }
            indices[atomNum - 2] = atomNum - 1;
            int min = 0;
            for (int i = 0; i < atomNum - 1; i++) {
                int min = 0;
                bool foundMin;
                do {
                    foundMin = true;
                    for (int j = 0; j < atomNum - 1; j++) {
                        if (indices[j + i] == min) {
                            foundMin = false;
                            min++;
                        }
                    }
                } while (!foundMin);
                indices[atomNum - 1 + i] = min;
                /*for(int l = 0; l < i; l++) {
                    printf("  ");
                }
                for(int k = i; k < atomNum-1+i; k++) {
                    printf("%d ", indices[k]);
                }*/
                atoms[indices[i]]->children.push_back(atoms[min]);
                //printf("\t%d -> %d\n", indices[i], min);
                atoms[min]->parent = atoms[indices[i]];
                res.edgeVertices.push_back(Vertex(vec2(atoms[indices[i]]->vertex.pos.x, atoms[indices[i]]->vertex.pos.y), lineColor));
                res.edgeVertices.push_back(Vertex(vec2(atoms[min]->vertex.pos.x, atoms[min]->vertex.pos.y), lineColor));
            }
        }
        else {
            for(int i = 0; i < atomNum-1; i++) {
                atoms[i]->children.push_back(atoms[i+1]);
                //printf("%d -> %d\n", i, i+1);
                atoms[i+1]->parent = atoms[i];
                res.edgeVertices.push_back(Vertex(vec2(atoms[i]->vertex.pos.x, atoms[i]->vertex.pos.y), lineColor));
                res.edgeVertices.push_back(Vertex(vec2(atoms[i + 1]->vertex.pos.x, atoms[i + 1]->vertex.pos.y), lineColor));
            }
        }
        res.root = atoms[0];
        //printf("xd\n", atomNum);
    }
    catch (std::bad_alloc & exc) {
        printf("hmm");
    }
}

void generateCircleVertices(Molecule& mol) {
    mol.cirVertices = std::vector<Vertex>();
    std::vector<Vertex> centres = mol.vertices;
    for(int i = 0; i < centres.size(); i++) {
        vec3 centreCol = centres[i].col;
        //printf("%f %f %f\n", centreCol.x, centreCol.y, centreCol.z);
        vec2 centrePos = centres[i].pos;
        mol.cirVertices.push_back(Vertex(centrePos, centreCol));
        for(int n = 0; n <= cRes; n++) {
            vec2 delta = vec2(sinf(2.0f*M_PI*(float)n/cRes)*atomRad,cosf(2.0f*M_PI*(float)n/cRes)*atomRad);
            Vertex v = Vertex(centrePos+delta,centreCol);
            mol.cirVertices.push_back(v);
            }
    }
}
void InterpolateLines(Molecule& mol) {
    std::vector<Vertex> temp = std::vector<Vertex>();
    for(int i = 0; i < mol.edgeVertices.size()/2; i++) {
        Vertex v1 = mol.edgeVertices[2 * i];
        Vertex v2 = mol.edgeVertices[2 * i + 1];
        for(int n = 0; n <= lRes; n++) {
            //vec3 lineColor = vec3((float)n/lRes,0.5*(float)n/lRes,1-(float)n/lRes);
            Vertex resV1 = Vertex((float) n/lRes*v2.pos + (1-(float)n/lRes)*v1.pos,lineColor);
            //Vertex resV2 = Vertex((float) (n+1)/lRes*v2.pos + (1-(float)(n+1)/lRes)*v1.pos,lineColor);
            //temp.push_back(resV2);
            temp.push_back(resV1);
        }
    }
    mol.edgeVertices = temp;
}
void generatePlate() {
    vec2 centrePos = vec2(0,0);
    plate.push_back(Vertex(centrePos, circleColor));
    for(int n = 0; n <= pRes; n++) {
        vec2 delta = vec2(sinf(2.0f*M_PI*(float)n/pRes),cosf(2.0f*M_PI*(float)n/pRes));
        Vertex v = Vertex(centrePos+delta,circleColor);
        plate.push_back(v);
        //printf("%f %f\n",v.pos.x,v.pos.y);
    }
}
void reGenVertices() {
    verticesVec = std::vector<Vertex>();
    Molecule* mol1 = mols[0];
    //mol1->vel=vec2(-1.0,-1.0);
    Molecule* mol2 = mols[1];
    verticesVec.insert(verticesVec.end(), plate.begin(),plate.end());
    verticesVec.insert(verticesVec.end(), mol1->edgeVertices.begin(), mol1->edgeVertices.end());
    verticesVec.insert(verticesVec.end(), mol1->cirVertices.begin(), mol1->cirVertices.end());
    verticesVec.insert(verticesVec.end(), mol2->edgeVertices.begin(), mol2->edgeVertices.end());
    verticesVec.insert(verticesVec.end(), mol2->cirVertices.begin(), mol2->cirVertices.end());

    vertices = verticesVec.data();
}

void newAtoms() {
    mols = std::vector<Molecule*>();
    verticesVec = std::vector<Vertex>();
    verticesVec.insert(verticesVec.end(), plate.begin(),plate.end());

    Molecule* mol1 = new Molecule();
    generateMolecule(*mol1);
    //mol1->vel = vec2(-1.0,-1.0);
    generateCircleVertices(*mol1);
    InterpolateLines(*mol1);
    mol1->move(vec2(2,2));
    mol1->calcMOI();
    mols.push_back(mol1);

    verticesVec.insert(verticesVec.end(), mol1->edgeVertices.begin(), mol1->edgeVertices.end());
    verticesVec.insert(verticesVec.end(), mol1->cirVertices.begin(), mol1->cirVertices.end());

    Molecule* mol2 = new Molecule();
    generateMolecule(*mol2);

    generateCircleVertices(*mol2);
    InterpolateLines(*mol2);
    mol2->move(vec2(-1,-1));
    mol2->calcMOI();
    mols.push_back(mol2);


    verticesVec.insert(verticesVec.end(), mol2->edgeVertices.begin(), mol2->edgeVertices.end());
    verticesVec.insert(verticesVec.end(), mol2->cirVertices.begin(), mol2->cirVertices.end());


    vertices = verticesVec.data();

}
vec2 calcCoulombForce(Atom a1, Atom a2, vec2 d) {
    vec2 res = k*normalize(d)*((a1.charge*a2.charge*pow(cE,2))/pow(length(d)*distC,2));
    printf("%f %f C\n", res.x, res.y);
    return res;
}
vec2 calcAirResistance(Atom a1, vec2 vel, float aVel, vec2 o) {
    vec2 res = (o*aVel+vel)*distC*vW*-1.0;
    //printf("xd");
    printf("%f %f R\n", res.x, res.y);
    return res;
}
float calcTorque(vec2 v, vec2 F) {
    vec3 res =  cross(v*distC,F);
    return res.z;
}
vec2 getParallel(vec2 v1, vec2 v2) {

}

vec2 getPerpendicular(vec2 v1, vec2 v2) {

}
void Physics(float dT) {
    std::vector<vec2> dV = std::vector<vec2>();
    std::vector<float> dO = std::vector<float>();
    for(int i = 0; i < mols.size(); i++) {
        vec2 sumForceMove = vec2(0,0);
        vec2 sumForceRot = vec2(0,0);
        float sumTorque = 0;
        Molecule mol = *mols[i];
        for(int j = 0; j < mol.atoms.size(); j++) {
            Atom a = *mol.atoms[j];
            vec2 sumForce = vec2(0,0);
            float sumTorque = 0;
            //printf("airrse!!!!");
            for(int k = 0; k < mols.size(); k++) {
                if(i==k) {
                    continue;
                }
                else {
                    Molecule molIn = (*mols[k]);
                    for(int l = 0; l < molIn.atoms.size(); l++) {
                      Atom aIn = *molIn.atoms[l];
                      vec2 d = a.relPos(aIn);
                      //printf("%f %f\n",d.x, d.y);
                      sumForce = sumForce + calcCoulombForce(a, aIn, d);
                      //10.0*vec2(2*(i%2)-1,2*(i%2)-1)
                    }
                }
            }
            sumForce = sumForce + calcAirResistance(a, mol.vel, mol.aVel, mol.relPos(a));   //TODO calc moving and rotating parts of force
            //sumForceMove += sumForce mozgató
            //sumTorque = sumTorque forgató + atom pozi;
            //printf("%f\n", sumTorque);
        }

        dV.push_back(sumForceMove/(mH*mol.mass)*dT);
        dO.push_back(sumTorque/(mH*mol.MOI)*dT);
        dO.push_back(sumTorque/(mH*mol.MOI)*dT);
        printf("%f F\n", dO.back());
    }
    for(int i = 0; i < dV.size(); i++) {
        Molecule* mol = mols[i];
        mol->move(vec2(mol->vel/distC*dT));
        mol->vel=mol->vel+dV[i];
        mol->rot(mol->aVel*dT);
        mol->aVel=mol->aVel+dO[i];
        //mol->move(vec2(1.0,1.0));

        //printf("%f %f %f %f\n", mol.vel.x, mol.vel.y, dV[i].x, dV[i].y);

        //mol.rot(mol.aVel*dT);
       // mol.aVel=mol.aVel+dO[i];
    }
}



// Initialization, create an OpenGL context
void onInitialization() {
    generatePlate();
    newAtoms();
    glViewport(0, 0, 600, 600); //windowWidth

	glGenVertexArrays(1, &vao);	// get 1 vao id
	glBindVertexArray(vao);		// make it active

	unsigned int vbo;		// vertex buffer object
	glGenBuffers(1, &vbo);	// Generate 1 buffer
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

    int size = verticesVec.size();

	glBufferData(GL_ARRAY_BUFFER,
                 // Copy to GPU target
		sizeof(float)*5*(size),//+plate.size()), //sizeof(mol->numEdges*sizeof(float)*4),  // # bytes
		vertices,	      	// address
		GL_DYNAMIC_DRAW);	// we do not change later

	glEnableVertexAttribArray(0);  // AttribArray 0
	glVertexAttribPointer(0,       // vbo -> AttribArray 0
		2, GL_FLOAT, GL_FALSE, // two floats/attrib, not fixed-point
		sizeof(Vertex), (void*) offsetof(Vertex, pos)); 		     // stride, offset: tightly packed

    glEnableVertexAttribArray(1);  // AttribArray 0
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, col));


    // create program for the GPU
	worldProgram.create(vertexSourceWorld, fragmentSource, "outColor");
    circleProgram.create(vertexSourceCircle, fragmentSource, "outColor");
    worldProgram.Use();
}

// Window has become invalid: Redraw
void onDisplay() {
    int size = verticesVec.size();
    glBufferData(GL_ARRAY_BUFFER, 	// Copy to GPU target
                 sizeof(float)*5*(size),//+plate.size()), //sizeof(mol->numEdges*sizeof(float)*4),  // # bytes
                 vertices,	      	// address
                 GL_DYNAMIC_DRAW);	// we do not change later*/
	glClearColor(bgrColor.x, bgrColor.y, bgrColor.z, 0);     // background color
	glClear(GL_COLOR_BUFFER_BIT); // clear frame buffer

	// Set color to (0, 1, 0) = green
	int location = glGetUniformLocation(worldProgram.getId(), "color");
	glUniform3f(location, 0.0f, 0.8f, 0.8f); // 3 floats

	float MVPtransf[4][4] = { 1, 0, 0, 0,    // MVP matrix,
							  0, 1, 0, 0,    // row-major!
							  0, 0, 1, 0,
                              xOffs, yOffs, 1, 1 };

	location = glGetUniformLocation(worldProgram.getId(), "MVP");	// Get the GPU location of uniform variable MVP
	glUniformMatrix4fv(location, 1, GL_TRUE, &MVPtransf[0][0]);	// Load a 4x4 row-major float matrix to the specified location

	glBindVertexArray(vao);  // Draw call

    //glDisable(GL_DEPTH_TEST);
    circleProgram.Use();
    glDrawArrays(GL_TRIANGLE_FAN, 0, plate.size());
    worldProgram.Use();
    int sizeSoFar = plate.size();
    for(int n = 0; n < mols.size(); n++) {
        if (drawLines) {
            for (int i = 0; i < mols[n]->edgeVertices.size()/(lRes+1); i++) {
                glDrawArrays(GL_LINE_STRIP, sizeSoFar + i * (lRes+1) /*startIdx*/, lRes+1 /*# Elements*/); //plate.size()
            }
        }
        if (drawCircles) {
            for (int i = 0; i < mols[n]->vertices.size(); i++) {
                glDrawArrays(GL_TRIANGLE_FAN, sizeSoFar + mols[n]->edgeVertices.size() + i * (cRes+2)  /*startIdx*/,
                             cRes+2  /*# Elements*/); //plate.size()+
            }
        }
        sizeSoFar += mols[n]->edgeVertices.size()+mols[n]->cirVertices.size();

    }

	glutSwapBuffers(); // exchange buffers for double buffering
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'o') drawLines = !drawLines;         // if d, invalidate display, i.e. redraw
    if (key == 'l') drawCircles = !drawCircles;
    if (key == 'e') yOffs+=0.1;
    if (key == 'x') yOffs-=0.1;
    if (key == 'd') xOffs+=0.1;
    if (key == 's') xOffs-=0.1;
    if (key == ' ') {
        newAtoms();
    }
    //if (key == 'f') {Physics(1.0);}//Physics();

    reGenVertices();
    glutPostRedisplay();
}


// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {	// pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;
	//printf("Mouse moved to (%3.2f, %3.2f)\n", cX, cY);
}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) { // pX, pY are the pixel coordinates of the cursor in the coordinate system of the operation system
	// Convert to normalized device space
	float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
	float cY = 1.0f - 2.0f * pY / windowHeight;

	char * buttonStat;
	switch (state) {
	case GLUT_DOWN: buttonStat = "pressed"; break;
	case GLUT_UP:   buttonStat = "released"; break;
	}
/*
	switch (button) {
	case GLUT_LEFT_BUTTON:   printf("Left button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);   break;
	case GLUT_MIDDLE_BUTTON: printf("Middle button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY); break;
	case GLUT_RIGHT_BUTTON:  printf("Right button %s at (%3.2f, %3.2f)\n", buttonStat, cX, cY);  break;
	}
*/
 }

// Idle event indicating that some time elapsed: do animation here
long prevTime = 0;
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
    float dT = (time-prevTime)/1000.0f;
    //printf("%f\n", dT);
    prevTime = time;
    Physics((float)dT);
    reGenVertices();
    glutPostRedisplay();
}
