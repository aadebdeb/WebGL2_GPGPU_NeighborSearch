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

  function createVbo(gl, array) {
    const vbo = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(gl.ARRAY_BUFFER, array, gl.STATIC_DRAW);
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

  function createVao(gl, vboObjs, ibo) {
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    if (ibo !== undefined) {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    }
    vboObjs.forEach((vboObj) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, vboObj.buffer);
      gl.enableVertexAttribArray(vboObj.index);
      gl.vertexAttribPointer(vboObj.index, vboObj.size, gl.FLOAT, false, 0, 0);
    });
    gl.bindVertexArray(null);
    if (ibo !== undefined) {
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    return vao;
  }

  function createTexture(gl, size, internalFormat, format) {
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, size, size, 0, format, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return texture;
  }

  function getUniformLocations(gl, program, keys) {
    const locations = {};
    keys.forEach(key => {
        locations[key] = gl.getUniformLocation(program, key);
    });
    return locations;
  }

  function setTextureAsUniform(gl, index, texture, location) {
    gl.activeTexture(gl.TEXTURE0 + index);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(location, index);
  }

  const INITIALIZE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out float o_value;

uniform vec2 u_randomSeed;

float random(vec2 x){
  return fract(sin(dot(x,vec2(12.9898, 78.233))) * 43758.5453);
}

void main(void) {
  o_value = random(gl_FragCoord.xy * 0.01 + u_randomSeed);
}
`

  const KERNEL_FRAGMENT_SHADER_SOURCE = 
`#version 300 es

precision highp float;

out float o_value;

uniform sampler2D u_valueTexture;
uniform uint u_size;
uniform uint u_blockStep;
uniform uint u_subBlockStep;

uint convertCoordToIndex(uvec2 coord) {
  return coord.x + coord.y * u_size;
}

uvec2 convertIndexToCoord(uint index) {
  return uvec2(index % u_size, index / u_size);
}

void main(void) {
  uint index = convertCoordToIndex(uvec2(gl_FragCoord.xy));
  uint d = 1u << (u_blockStep - u_subBlockStep);

  bool up = ((index >> u_blockStep) & 2u) == 0u;

  uint targetIndex;
  if ((index & d) == 0u) {
    targetIndex = index | d;
  } else {
    targetIndex = index & ~d;
    up = !up;
  }

  float a = texelFetch(u_valueTexture, ivec2(gl_FragCoord.xy), 0).x;
  float b = texelFetch(u_valueTexture, ivec2(convertIndexToCoord(targetIndex)), 0).x;
  if ((a > b) == up) {
    o_value = b; // swap
  } else {
    o_value = a; // no_swap
  }
}

`;

const FILL_SCREEN_VERTEX_SHADER_SOURCE =
`#version 300 es

layout (location = 0) in vec2 position;

void main(void) {
  gl_Position = vec4(position, 0.0, 1.0);
}
`

  const INITIALIZE_PARTICLE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

layout (location = 0) out vec2 o_position;
layout (location = 1) out vec2 o_velocity;

uniform vec2 u_randomSeed;

float random(vec2 x){
  return fract(sin(dot(x,vec2(12.9898, 78.233))) * 43758.5453);
}

void main(void) {
  o_position = vec2(
    random(gl_FragCoord.xy * 0.013 + u_randomSeed + vec2(32.19, 27.51)),
    random(gl_FragCoord.xy * 0.029 + u_randomSeed + vec2(19.56, 11.34))
  );
  o_velocity = (vec2(
    random(gl_FragCoord.xy * 0.059 + u_randomSeed + vec2(27.31, 16.91)),
    random(gl_FragCoord.xy * 0.038 + u_randomSeed + vec2(25.95, 19.47))
  ) * 2.0 - 1.0) * 0.2;
}
`

  const UPDATE_PARTICLE_FRAGMENT_SHADER_SOURCE = 
`#version 300 es

precision highp float;

layout (location = 0) out vec2 o_position;
layout (location = 1) out vec2 o_velocity;

uniform sampler2D u_positionTexture;
uniform sampler2D u_velocityTexture;
uniform float u_deltaTime;

void main(void) {
  ivec2 coord = ivec2(gl_FragCoord.xy);
  vec2 position = texelFetch(u_positionTexture, coord, 0).xy;
  vec2 velocity = texelFetch(u_velocityTexture, coord, 0).xy;

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

  const RENDER_PARTICLE_VERTEX_SHADER_SOURCE =
`#version 300 es

layout (location = 0) in float o_index;

uniform sampler2D u_positionTexture;
uniform vec2 u_canvasSize;

ivec2 convertIndexToCoord(int index, ivec2 size) {
  return ivec2(index % size.x, index / size.x);
}

void main(void) {
  vec2 scale = min(u_canvasSize.x, u_canvasSize.y) / u_canvasSize;
  ivec2 coord = convertIndexToCoord(int(o_index), textureSize(u_positionTexture, 0));
  gl_Position = vec4(scale * (texelFetch(u_positionTexture, coord, 0).xy * 2.0 - 1.0), 0.0, 1.0);
  gl_PointSize = 5.0;
}
`

  const RENDER_PARTICLE_FRAGMENT_SHADER_SOURCE =
`#version 300 es

precision highp float;

out vec4 o_color;

void main(void) {
  if (length(gl_PointCoord - 0.5) < 0.5) {
    o_color = vec4(1.0);
  } else {
    discard;
  }
}
`

  const VERTICES_POSITION = new Float32Array([
    -1.0, -1.0,
    1.0, -1.0,
    -1.0,  1.0,
    1.0,  1.0
  ]);

  const VERTICES_INDEX = new Int16Array([
    0, 1, 2,
    3, 2, 1
  ]);


  function createInitializeParticlesProgram(gl) {
    const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, INITIALIZE_PARTICLE_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  function createUpdateParticlesProgram(gl) {
    const vertexShader = createShader(gl, FILL_SCREEN_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, UPDATE_PARTICLE_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  function createRenderParticlesProgram(gl) {
    const vertexShader = createShader(gl, RENDER_PARTICLE_VERTEX_SHADER_SOURCE, gl.VERTEX_SHADER);
    const fragmentShader = createShader(gl, RENDER_PARTICLE_FRAGMENT_SHADER_SOURCE, gl.FRAGMENT_SHADER);
    return createProgram(gl, vertexShader, fragmentShader);
  }

  function createParticleFramebuffer(gl, size) {
    const framebuffer = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    const positionTexture = createTexture(gl, size, gl.RG32F, gl.RG);
    gl.bindTexture(gl.TEXTURE_2D, positionTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, positionTexture, 0);
    const velocityTexture = createTexture(gl, size, gl.RG32F, gl.RG);
    gl.bindTexture(gl.TEXTURE_2D, velocityTexture);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, velocityTexture, 0);
    gl.drawBuffers([gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.bindTexture(gl.TEXTURE_2D, null);
    return {
      framebuffer: framebuffer,
      positionTexture: positionTexture,
      velocityTexture: velocityTexture
    };
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

  const initializeParticleProgram = createInitializeParticlesProgram(gl);
  const updateParticleProgram = createUpdateParticlesProgram(gl);
  const renderParticleProgram = createRenderParticlesProgram(gl);

  const initializeParticleUniforms = getUniformLocations(gl, initializeParticleProgram, ['u_randomSeed']);
  const updateParticleUniforms = getUniformLocations(gl, updateParticleProgram, ['u_positionTexture', 'u_velocityTexture', 'u_deltaTime']);
  const renderParticleUniforms = getUniformLocations(gl, renderParticleProgram, ['u_positionTexture', 'u_canvasSize']);

  const fillScreenVao = createVao(gl,
    [{
      buffer: createVbo(gl, VERTICES_POSITION),
      size: 2,
      index: 0
    }],
    createIbo(gl, VERTICES_INDEX)
  );

  const particleTextureSize = 128;
  const particleNum = particleTextureSize * particleTextureSize;
  const particleVao = createVao(gl,[{
    buffer: createVbo(gl, new Float32Array(Array.from({length: particleNum}, (v, i) => i))),
    size: 1,
    index: 0
  }]);

  let particleFbObjR = createParticleFramebuffer(gl, particleTextureSize);
  let particleFbObjW = createParticleFramebuffer(gl, particleTextureSize);
  const swapParticleFbObj = function() {
    const tmp = particleFbObjR;
    particleFbObjR = particleFbObjW;
    particleFbObjW = tmp;
  };

  const initializeParticles = function() {
    gl.bindFramebuffer(gl.FRAMEBUFFER, particleFbObjW.framebuffer);
    gl.viewport(0.0, 0.0, particleTextureSize, particleTextureSize);

    gl.useProgram(initializeParticleProgram);
    gl.uniform2f(initializeParticleUniforms['u_randomSeed'], Math.random() * 100.0, Math.random() * 100.0);

    gl.bindVertexArray(fillScreenVao);
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    gl.bindVertexArray(null);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    swapParticleFbObj();
  };

  const updateParticles = function(deltaTime) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, particleFbObjW.framebuffer);
    gl.viewport(0.0, 0.0, particleTextureSize, particleTextureSize);

    gl.useProgram(updateParticleProgram);
    setTextureAsUniform(gl, 0, particleFbObjR.positionTexture, updateParticleUniforms['u_positionTexture']);
    setTextureAsUniform(gl, 1, particleFbObjR.velocityTexture, updateParticleUniforms['u_velocityTexture']);
    gl.uniform1f(updateParticleUniforms['u_deltaTime'], deltaTime);

    gl.bindVertexArray(fillScreenVao);
    gl.drawElements(gl.TRIANGLES, 6, gl.UNSIGNED_SHORT, 0);
    gl.bindVertexArray(null);

    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    swapParticleFbObj();
  };


  initializeParticles();

  gl.clearColor(0.2, 0.2, 0.2, 1.0);
  let previousTime = performance.now();
  const render = function() {
    const currentTime = performance.now();
    const deltaTime = Math.min(0.05, (currentTime - previousTime) * 0.001);
    previousTime = currentTime;

    updateParticles(deltaTime);

    gl.viewport(0.0, 0.0, canvas.width, canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(renderParticleProgram);
    setTextureAsUniform(gl, 0, particleFbObjR.positionTexture, renderParticleUniforms['u_positionTexture']);
    gl.uniform2f(renderParticleUniforms['u_canvasSize'], canvas.width, canvas.height);

    gl.bindVertexArray(particleVao);
    gl.drawArrays(gl.POINTS, 0, particleNum);
    gl.bindVertexArray(null);

    requestAnimationFrame(render);
  };
  render();

}());