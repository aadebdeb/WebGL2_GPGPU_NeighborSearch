(function() {

  function createShader(gl, source, type) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
        throw new Error(gl.getShaderInfoLog(shader) + source);
    }
    return shader;
  }

  function createProgram(gl, vertexShader, fragmentShader) {
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        throw new Error(gl.getProgramInfoLog(program));
    }
    return program;
  }

  function createTransformFeedbackProgram(gl, vertexShader, fragmentShader, varyings) {
    const program = gl.createProgram();
    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.transformFeedbackVaryings(program, varyings, gl.SEPARATE_ATTRIBS);
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(program));
  }
  return program;
  }

  function createVbo(gl, array, usage) {
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, array, usage !== undefined ? usage : gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return vbo;
  }

  function createIbo(gl, array) {
    const ibo = gl.createBuffer();
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, array, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    return ibo;
  }

  function getUniformLocations(gl, program, keys) {
    const locations = {};
    keys.forEach(key => {
        locations[key] = gl.getUniformLocation(program, key);
    });
    return locations;
  }

  function addVertex3(vertices, vi, x, y, z) {
    vertices[vi++] = x;
    vertices[vi++] = y;
    vertices[vi++] = z;
    return vi;
  };

  function addTriangle(indices, i, v0, v1, v2) {
    indices[i++] = v0;
    indices[i++] = v1;
    indices[i++] = v2;
    return i;
  };

  function addQuad(indices, i, v00, v10, v01, v11) {
    indices[i] = v00;
    indices[i + 1] = indices[i + 5] = v10;
    indices[i + 2] = indices[i + 4] = v01;
    indices[i + 3] = v11;
    return i + 6;
  };

  function createSphere(radius, thetaSegment, phiSegment) {
    const vertexNum = 2 + (thetaSegment - 1) * phiSegment;
    const indexNum = phiSegment * 6 + (thetaSegment - 2) * phiSegment * 6;
    const indices = new Int16Array(indexNum);
    const positions = new Float32Array(3 * vertexNum);
    const normals = new Float32Array(3 * vertexNum);

    const thetaStep = Math.PI / thetaSegment;
    const phiStep = 2.0 * Math.PI / phiSegment;

    // setup positions & normals
    let posCount = 0;
    let normalCount = 0;
    posCount = addVertex3(positions, posCount, 0, -radius, 0);
    normalCount = addVertex3(normals, normalCount, 0, -1, 0);
    for (let hi = 1; hi < thetaSegment; hi++) {
      const theta = Math.PI - hi * thetaStep;
      const sinT = Math.sin(theta);
      const cosT = Math.cos(theta);
      for (let pi = 0; pi < phiSegment; pi++) {
        const phi = pi * phiStep;
        const sinP = Math.sin(-phi);
        const cosP = Math.cos(-phi);
        const p = new Vector3(
          radius * sinT * cosP,
          radius * cosT,
          radius * sinT * sinP
        );
        posCount = addVertex3(positions, posCount, p.x, p.y, p.z);
        const np = Vector3.norm(p);
        normalCount = addVertex3(normals, normalCount, np.x, np.y, np.z);
      }
    }
    posCount = addVertex3(positions, posCount, 0, radius, 0);
    normalCount = addVertex3(normals, normalCount, 0, 1, 0);

    // setup indices
    let indexCount = 0;
    for (let pi = 0; pi < phiSegment; pi++) {
      indexCount = addTriangle(indices, indexCount, 0, pi !== phiSegment - 1 ? pi + 2 : 1, pi + 1);
    }
    for (let hi = 0; hi < thetaSegment - 2; hi++) {
      const hj = hi + 1;
      for (let pi = 0; pi < phiSegment; pi++) {
        const pj = pi !== phiSegment - 1 ? pi + 1 : 0;
        indexCount = addQuad(indices, indexCount, 
          pi + hi * phiSegment + 1,
          pj + hi * phiSegment + 1,
          pi + hj * phiSegment + 1,
          pj + hj * phiSegment + 1
        );
      }
    }
    for (let pi = 0; pi < phiSegment; pi++) {
      indexCount = addTriangle(indices, indexCount,
        vertexNum - 1,
        pi + (thetaSegment - 2) * phiSegment + 1,
        (pi !== phiSegment - 1 ? pi + 1 : 0) + (thetaSegment - 2) * phiSegment + 1
      );
    }

    return {
      indices: indices,
      positions: positions,
      normals: normals,
    };
  }

  const UPDATE_PARTICLE_VERTEX_SHADER_SOURCE = 
`#version 300 es

precision highp float;

layout (location = 0) in vec3 i_position;
layout (location = 1) in vec3 i_velocity;

out vec3 o_position;
out vec3 o_velocity;

uniform float u_deltaTime;

void main(void) {
  vec3 position = i_position;
  vec3 velocity = i_velocity;

  vec3 nextPosition = position + u_deltaTime * velocity;
  vec3 nextVelocity = velocity;

  if (nextPosition.x <= 0.001) {
    nextVelocity.x *= -1.0;
    nextPosition.x += u_deltaTime * nextVelocity.x;
  }
  if (nextPosition.x >= 0.999) {
    nextVelocity.x *= -1.0;
    nextPosition.x += u_deltaTime * nextVelocity.x;
  }
  if (nextPosition.y <= 0.001) {
    nextVelocity.y *= -1.0;
    nextPosition.y += u_deltaTime * nextVelocity.y;
  }
  if (nextPosition.y >= 0.999) {
    nextVelocity.y *= -1.0;
    nextPosition.y += u_deltaTime * nextVelocity.y;
  }
  if (nextPosition.z <= 0.001) {
    nextVelocity.z *= -1.0;
    nextPosition.z += u_deltaTime * nextVelocity.z;
  }
  if (nextPosition.z >= 0.999) {
    nextVelocity.z *= -1.0;
    nextPosition.z += u_deltaTime * nextVelocity.z;
  }

  o_position = nextPosition;
  o_velocity = nextVelocity;
}
`

  const TRANSFORM_FEEDBAK_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out vec4 o_color;

void main(void) {
  o_color = vec4(1.0);
}
`

  const RENDER_PARTICLE_VERTEX_SHADER_SOURCE =
`#version 300 es

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in float value;
layout (location = 3) in vec3 instancePosition;

out vec3 v_color;
out vec3 v_normal;

uniform mat4 u_mvpMatrix;
uniform float u_maxValue;

vec3[7] HEAT_MAP_COLORS = vec3[7](
  vec3(0.5),
  vec3(0.0, 0.0, 1.0),
  vec3(0.0, 0.5, 0.5),
  vec3(0.0, 1.0, 0.0),
  vec3(0.5, 0.5, 0.0),
  vec3(0.8, 0.4, 0.0),
  vec3(1.0, 0.0, 0.0)
);

vec3 getHeatmapColor(float value, float maxValue) {
  float step = maxValue / 7.0;
  float i = min(value, maxValue - 0.0001) / step;
  return mix(HEAT_MAP_COLORS[int(i)], HEAT_MAP_COLORS[int(i) + 1], fract(i));
}

void main(void) {
  v_color = getHeatmapColor(value, u_maxValue);
  v_normal = (u_mvpMatrix * vec4(normal, 0.0)).xyz;
  vec3 pos = position + (2.0 * instancePosition - 1.0) * 500.0;
  gl_Position = u_mvpMatrix * vec4(pos, 1.0);
}
`

  const RENDER_PARTICLE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

in vec3 v_color;
in vec3 v_normal;

out vec4 o_color;

vec3 LightDir = normalize(vec3(1.0, 1.0, -1.0));

void main(void) {
  vec3 normal = normalize(v_normal);
  float dotNL = dot(normal, LightDir);
  o_color = vec4(v_color * smoothstep(-1.0, 1.0, dotNL), 1.0);
}
`

  function createUpdateParticleProgram(gl) {
    const vertexShader = createShader(gl, UPDATE_PARTICLE_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, TRANSFORM_FEEDBAK_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createTransformFeedbackProgram(gl, vertexShader, fragmentShader, ['o_position', 'o_velocity']);
  }

  function createRenderParticleProgram(gl) {
    const vertexShader = createShader(gl, RENDER_PARTICLE_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, RENDER_PARTICLE_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  const canvas = document.getElementById('canvas');
  const resizeCanvas = function() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  };
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  const gl = canvas.getContext('webgl2');
  gl.getExtension('EXT_color_buffer_float');

  const updateParticleProgram = createUpdateParticleProgram(gl);
  const renderParticleProgram = createRenderParticleProgram(gl);

  const updateParticleUniforms = getUniformLocations(gl, updateParticleProgram, ['u_deltaTime']);
  const renderParticleUniforms = getUniformLocations(gl, renderParticleProgram, ['u_mvpMatrix', 'u_maxValue']);

  const stats = new Stats();
  document.body.appendChild(stats.dom);

  const gui = new dat.GUI();
  const data = {
    'N': 5,
    'particle num': 2 ** (5 * 2),
    'view radius': 0.1,
    'max value': 300,
    sphere: {
      'radius': 5.0,
      'theta segment': 16,
      'phi segment': 8
    },
    camera: {
      'angle': -45.0,
      'distance': 1000.0,
      'height': 1000.0
    },
    'reset': () => reset()
  };
  gui.add(data, 'N', 1, 8).step(1).onChange((v) => {
    data['particle num'] = 2 ** (v * 2);
  });
  gui.add(data, 'particle num').listen();
  gui.add(data, 'view radius', 0.05, 0.25);
  gui.add(data, 'max value', 0, 1000);
  const guiSphere = gui.addFolder('sphere');
  guiSphere.add(data.sphere, 'radius', 1.0, 20.0);
  guiSphere.add(data.sphere, 'theta segment', 4, 64).step(1);
  guiSphere.add(data.sphere, 'phi segment', 4, 64).step(1);
  const guiCamera = gui.addFolder('camera');
  guiCamera.add(data.camera, 'angle', -180, 180);
  guiCamera.add(data.camera, 'distance', 50.0, 3000.0);
  guiCamera.add(data.camera, 'height', -3000.0, 3000.0);
  gui.add(data, 'reset');

  let requestId = null;
  function reset() {
    if (requestId !== null) {
      cancelAnimationFrame(requestId);
    }

    const sphere = createSphere(data.sphere['radius'], data.sphere['theta segment'], data.sphere['phi segment']);
    const sphereIbo = createIbo(gl, sphere.indices);
    const vertexPositionVbo = createVbo(gl, sphere.positions);
    const vertexNormalVbo = createVbo(gl, sphere.normals);

    const particleTextureSizeN = data['N'];
    const particleTextureSize = 2 ** particleTextureSizeN;
    const particleNum = particleTextureSize * particleTextureSize;

    let positions = new Float32Array(Array.from({length: particleNum * 3}, () => {
      return Math.sqrt(Math.random());
    }));
    const velocities = new Float32Array(Array.from({length: particleNum * 3}, () => {
      return (Math.random() * 2.0 - 1.0) * 0.05;
    }));
    const values = new Float32Array(particleNum);

    let positionVboR = createVbo(gl, positions, gl.DYNAMIC_COPY);
    let velocityVboR = createVbo(gl, velocities, gl.DYNAMIC_COPY);
    let positionVboW = createVbo(gl, new Float32Array(particleNum * 3), gl.DYNAMIC_COPY);
    let velocityVboW = createVbo(gl, new Float32Array(particleNum * 3), gl.DYNAMIC_COPY);
    const valueVbo = createVbo(gl, values);
    const swapParticleVbos = function() {
      const tmpP = positionVboR;
      const tmpV = velocityVboR;
      positionVboR = positionVboW;
      velocityVboR = velocityVboW;
      positionVboW = tmpP;
      velocityVboW = tmpV;
    }

    const viewRadius = data['view radius'];

    const transformFeedback = gl.createTransformFeedback();

    const computeValues = function() {
      gl.bindBuffer(gl.ARRAY_BUFFER, positionVboR);
      gl.getBufferSubData(gl.ARRAY_BUFFER, 0, positions);

      const bucketSize = 2.0 * viewRadius;
      const bucketNum = Math.ceil(1.0 / bucketSize);
      const buckets = new Array(bucketNum);
      for (let i = 0; i < bucketNum; i++) {
        buckets[i] = new Array(bucketNum);
        for (let j = 0; j < bucketNum; j++) {
          buckets[i][j] = [];
          for (let k = 0; k < bucketNum; k++) {
            buckets[i][j][k] = [];
          }
        }
      }
      for (let i = 0; i < particleNum; i++) {
        const idxX = Math.floor(positions[3 * i] / bucketSize);
        const idxY = Math.floor(positions[3 * i + 1] / bucketSize);
        const idxZ = Math.floor(positions[3 * i + 2] / bucketSize);
        buckets[idxX][idxY][idxZ].push(i);
      }

      const getNeighbors = function(idx, position, idxX, idxY, idxZ) {
        if (idxX < 0 || idxX >= bucketNum || idxY < 0 || idxY >= bucketNum || idxZ < 0 || idxZ >= bucketNum) {
          return 0.0;
        }
        return buckets[idxX][idxY][idxZ].reduce((value, i) => {
          if (i !== idx) {
            const otherPos = new Vector3(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]);
            const d = Vector3.dist(position, otherPos);
            if (d < viewRadius) {
              value += (viewRadius - d) / viewRadius;
            }
          }
          return value;
        }, 0.0);
      }

      for (let i = 0; i < particleNum; i++) {
        const position = new Vector3(positions[3 * i], positions[3 * i + 1], positions[3 * i + 2]);

        const idxX = Math.floor(position.x / bucketSize);
        const idxY = Math.floor(position.y / bucketSize);
        const idxZ = Math.floor(position.z / bucketSize);
        const offsetX = (position.x / bucketSize - idxX) < 0.5 ? -1 : 1;
        const offsetY = (position.y / bucketSize - idxY) < 0.5 ? -1 : 1;
        const offsetZ = (position.z / bucketSize - idxZ) < 0.5 ? -1 : 1;

        let value = 0.0;
        value += getNeighbors(i, position, idxX, idxY, idxZ);
        value += getNeighbors(i, position, idxX + offsetX, idxY, idxZ);
        value += getNeighbors(i, position, idxX, idxY + offsetY, idxZ);
        value += getNeighbors(i, position, idxX + offsetX, idxY + offsetY, idxZ);
        value += getNeighbors(i, position, idxX + offsetX, idxY, idxZ + offsetZ);
        value += getNeighbors(i, position, idxX, idxY + offsetY, idxZ + offsetZ);
        value += getNeighbors(i, position, idxX + offsetX, idxY + offsetY, idxZ + offsetZ);

        values[i] = value;
      }

      gl.bindBuffer(gl.ARRAY_BUFFER, valueVbo);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, values);
    }

    const updateParticles = function(deltaTime) {
      gl.useProgram(updateParticleProgram);
      gl.uniform1f(updateParticleUniforms['u_deltaTime'], deltaTime);
      [positionVboR, velocityVboR].forEach((vbo, i) => {
        gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
        gl.enableVertexAttribArray(i),
        gl.vertexAttribPointer(i, 3, gl.FLOAT, false, 0, 0);
      });
      gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, transformFeedback);
      gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, positionVboW);
      gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 1, velocityVboW);
      gl.enable(gl.RASTERIZER_DISCARD);
      gl.beginTransformFeedback(gl.POINTS);
      gl.drawArrays(gl.POINTS, 0, particleNum);
      gl.disable(gl.RASTERIZER_DISCARD);
      gl.endTransformFeedback();
      gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
      gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 1, null);
      swapParticleVbos();

      computeValues();
    };

    gl.clearColor(0.2, 0.2, 0.2, 1.0);
    let previousTime = performance.now();
    const render = function() {
      stats.update();
  
      const currentTime = performance.now();
      const deltaTime = Math.min(0.05, (currentTime - previousTime) * 0.001);
      previousTime = currentTime;

      const cameraRadian = Math.PI * data.camera['angle'] / 180.0;
      const viewMatrix = Matrix4.inverse(Matrix4.lookAt(
        new Vector3(data.camera['distance'] * Math.cos(cameraRadian), data.camera['height'], data.camera['distance'] * Math.sin(cameraRadian)),
        Vector3.zero, new Vector3(0.0, 1.0, 0.0)
      ));
      const projectionMatrix = Matrix4.perspective(canvas.width / canvas.height, 60, 0.01, 10000.0);
      const mvpMatrix = Matrix4.mul(viewMatrix, projectionMatrix);

      updateParticles(deltaTime);
  
      gl.viewport(0.0, 0.0, canvas.width, canvas.height);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.useProgram(renderParticleProgram);
      gl.uniform1f(renderParticleUniforms['u_maxValue'], data['max value']);
      gl.uniformMatrix4fv(renderParticleUniforms['u_mvpMatrix'], false, mvpMatrix.elements);
  
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, sphereIbo);
      [[vertexPositionVbo, 3], [vertexNormalVbo, 3], [valueVbo, 1], [positionVboR, 3]].forEach((obj, i) => {
        gl.bindBuffer(gl.ARRAY_BUFFER, obj[0]);
        gl.enableVertexAttribArray(i);
        gl.vertexAttribPointer(i, obj[1], gl.FLOAT, false, 0, 0);
      });
      gl.vertexAttribDivisor(2, 1);
      gl.vertexAttribDivisor(3, 1);
      gl.drawElementsInstanced(gl.TRIANGLES, sphere.indices.length, gl.UNSIGNED_SHORT, 0, particleNum);

      requestId = requestAnimationFrame(render);
    };
    render();
  };
  reset();
}());