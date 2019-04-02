(function() {

  const UPDATE_PARTICLE_VERTEX_SHADER_SOURCE = 
`#version 300 es

precision highp float;

layout (location = 0) in vec2 i_position;
layout (location = 1) in vec2 i_velocity;

out vec2 o_position;
out vec2 o_velocity;

uniform float u_deltaTime;

void main(void) {
  vec2 position = i_position;
  vec2 velocity = i_velocity;

  vec2 nextPosition = position + u_deltaTime * velocity;
  vec2 nextVelocity = velocity;

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

layout (location = 0) in vec2 position;
layout (location = 1) in float value;

out vec3 v_color;

uniform vec2 u_canvasSize;
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
  vec2 scale = min(u_canvasSize.x, u_canvasSize.y) / u_canvasSize;
  gl_Position = vec4(scale * (position * 2.0 - 1.0), 0.0, 1.0);
  gl_PointSize = 5.0;

  v_color = getHeatmapColor(value, u_maxValue);
}
`

  const RENDER_PARTICLE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

in vec3 v_color;

out vec4 o_color;

void main(void) {
  if (length(gl_PointCoord - 0.5) < 0.5) {
    o_color = vec4(v_color, 1.0);
  } else {
    discard;
  }
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
  const renderParticleUniforms = getUniformLocations(gl, renderParticleProgram, ['u_canvasSize', 'u_viewRadius', 'u_maxValue']);

  const stats = new Stats();
  document.body.appendChild(stats.dom);

  const gui = new dat.GUI();
  const data = {
    'N': 6,
    'particle num': 2 ** (6 * 2),
    'view radius': 0.1,
    'max value': 300,
    'reset': () => reset()
  };
  gui.add(data, 'N', 1, 8).step(1).onChange((v) => {
    data['particle num'] = 2 ** (v * 2);
  });
  gui.add(data, 'particle num').listen();
  gui.add(data, 'view radius', 0.05, 0.25);
  gui.add(data, 'max value', 0, 1000);
  gui.add(data, 'reset');

  let requestId = null;
  function reset() {
    if (requestId !== null) {
      cancelAnimationFrame(requestId);
    }

    const particleTextureSizeN = data['N'];
    const particleTextureSize = 2 ** particleTextureSizeN;
    const particleNum = particleTextureSize * particleTextureSize;

    let positions = new Float32Array(Array.from({length: particleNum * 2}, () => {
      return Math.sqrt(Math.random());
    }));
    const velocities = new Float32Array(Array.from({length: particleNum * 2}, () => {
      return (Math.random() * 2.0 - 1.0) * 0.05;
    }));
    const values = new Float32Array(particleNum);

    let positionVboR = createVbo(gl, positions, gl.DYNAMIC_COPY);
    let velocityVboR = createVbo(gl, velocities, gl.DYNAMIC_COPY);
    let positionVboW = createVbo(gl, new Float32Array(particleNum * 2), gl.DYNAMIC_COPY);
    let velocityVboW = createVbo(gl, new Float32Array(particleNum * 2), gl.DYNAMIC_COPY);
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
        }
      }
      for (let i = 0; i < particleNum; i++) {
        const idxX = Math.floor(positions[2 * i] / bucketSize);
        const idxY = Math.floor(positions[2 * i + 1] / bucketSize);
        buckets[idxX][idxY].push(i);
      }

      const getNeighbors = function(idx, position, idxX, idxY) {
        if (idxX < 0 || idxX >= bucketNum || idxY < 0 || idxY >= bucketNum) {
          return 0.0;
        }
        return buckets[idxX][idxY].reduce((value, i) => {
          if (i !== idx) {
            const otherPos = new Vector2(positions[2 * i], positions[2 * i + 1]);
            const d = Vector2.dist(position, otherPos);
            if (d < viewRadius) {
              value += (viewRadius - d) / viewRadius;
            }
          }
          return value;
        }, 0.0);
      }

      for (let i = 0; i < particleNum; i++) {
        const position = new Vector2(positions[2 * i], positions[2 * i + 1]);

        const idxX = Math.floor(position.x / bucketSize);
        const idxY = Math.floor(position.y / bucketSize);
        const offsetX = (position.x / bucketSize - idxX) < 0.5 ? -1 : 1;
        const offsetY = (position.y / bucketSize - idxY) < 0.5 ? -1 : 1;

        let value = 0.0;
        value += getNeighbors(i, position, idxX, idxY);
        value += getNeighbors(i, position, idxX + offsetX, idxY);
        value += getNeighbors(i, position, idxX, idxY + offsetY);
        value += getNeighbors(i, position, idxX + offsetX, idxY + offsetY);

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
        gl.vertexAttribPointer(i, 2, gl.FLOAT, false, 0, 0);
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
  
      updateParticles(deltaTime);
  
      gl.viewport(0.0, 0.0, canvas.width, canvas.height);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.useProgram(renderParticleProgram);
      gl.uniform2f(renderParticleUniforms['u_canvasSize'], canvas.width, canvas.height);
      gl.uniform1f(renderParticleUniforms['u_viewRadius'], viewRadius);
      gl.uniform1f(renderParticleUniforms['u_maxValue'], data['max value']);

      [[positionVboR, 2], [valueVbo, 1]].forEach((obj, i) => {
        gl.bindBuffer(gl.ARRAY_BUFFER, obj[0]);
        gl.enableVertexAttribArray(i);
        gl.vertexAttribPointer(i, obj[1], gl.FLOAT, false, 0, 0);
      });
      gl.drawArrays(gl.POINTS, 0, particleNum);
      gl.bindVertexArray(null);

      requestId = requestAnimationFrame(render);
    };
    render();
  };
  reset();
}());