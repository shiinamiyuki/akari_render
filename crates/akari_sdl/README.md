# AkariRender Scene Description Language
A simple node-based scene description language.


## Syntax
### Built-in Types
The language supports the following built-in types:
`int, uint, float, bool, string, file, bytes, float2, float3, float4, int2, int3, int4, uint2, uint3, uint4,mat2, mat3, mat4`
Compound types:
- `[T]` - array/slice of T.
- `T?` - optional T
- `T1 | T2` - either T1 or T2

Special types:
- `any` - any type

Nodes and groups are not part of the type system, they are used to represent operations. So they cannot be used to describe data.

### Special Forms
- `@env("ENV_VAR")` - reads an environment variable as a string
- `@config("CONFIG_KEY")` - reads a configuration value as a string
- `@parse(string)` - parses a string as part of the document
- `@include("path/to/file.akari")` - include another file
- `@include_str("path/to/file.txt")` - include a text file as string. Equivalent to `@parse(@include("path/to/file.txt"))`
- `@include_bytes("path/to/file.bin")` - include a binary file as bytes
- `@base64("base64_string")` - decodes a base64 string
- `@from_bytes(bytes)` - construct values from bytes
- `@open("path/to/file.txt")` - creates a file handle. 
- `@error("message")` - emits an error message when there is error loading the field
- `@warning("message")` - emits a warning message when an optional field is not found
- `@doc("documentation")` - adds documentation to the field

### Raw String Literals
Raw string literals are enclosed in triple quotes `"""`. They can span multiple lines and contain escape sequences.

### Example
```rust
// enums are used to define a set of named constants
enum LightFlags : uint{
    Ambient = 1
    Directional = 2
    Point = 4
    Spot = 8
}
// enum can be any (non-reference) struct type
enum GoodVectors : float3 {
    Up = float3(0, 1, 0)
    Forward = float3(0, 0, 1)
    Right = float3(1, 0, 0)
}

type LightType = Point | Directional | Spot

// Defines a struct
// Structs are concrete types and can be put into buffers
struct Camera {
    position: float3
    target: float3
    up: float3
}

struct Mesh {
    vertices: [float3]
    indices: [uint]
    normals:  [float3]?
    texcoords: [float2]?
    material: ref Material? // reference to a material
    // structs can also be embedded in other structs
    embedded_material: Material?
}

opaque RenderPipeline {
    // opaque types are used to hide implementation details
    // they are used to represent resources that are not directly manipulated by the user
    // opaque types can also have struct fields
    enable_gi: bool
}

// Defines an operation node
// Nodes are abstract representations of operations
// node can have input and output sockets
// Sockets are of struct/opaque types
node LoadMesh {
    // input socket
    in filename: string = "mesh.obj"
    // output socket
    out mesh: Mesh 
}

struct Material {

}

// Defines a struct that extends another struct
struct DiffuseMaterial : Material {
    albedo: float3
}
struct SpecularMaterial : Material {
    albedo: float3
    roughness: float
}

struct MixedMaterial : Material {
    material1: ref Material
    material2: ref Material
    mix: float
}

// Defines a node group
// groups are similar to nodes but they can have internal nodes
// groups cannot have optional inputs. instead, they can have default values
group UberMaterial {
    in albedo: float3
    in roughness: float
    in metallic: float

    out material: MixedMaterial {
        material1: diffuse
        material2: specular
        mix: metallic
    }

    // internal nodes
    diffuse: DiffuseMaterial {
        albedo: albedo
    }
    // the order of the nodes does not matter
    specular: SpecularMaterial {
        albedo: albedo
        roughness: roughness
    }
    
}

struct Scene {
    camera: Camera
    meshes: [Mesh]
    materials: [Material]
    render_pipeline: RenderPipeline
}

// Constants
const PI: float = 3.14159265359
const MyFavoriteCamera: Camera = Camera {
    position: float3(0, 0, 0)
    target: float3(0, 0, 1)
    up: float3(0, 1, 0)
}

// export defines the entry point of description
// export block is like a graph block but it does not have input sockets
export Scene {
    load_mesh: LoadMesh {
        filename: "mesh.obj"
    }
    // there can be only one out in export block
    out scene: Scene {
        camera: Camera {
            position: float3(0, 0, 0)
            target: float3(0, 0, 1)
            up: float3(0, 1, 0)
        }
        meshes: [
           load_mesh.mesh
        ]
        materials: [
            UberMaterial {
                albedo: float3(1, 1, 1)
                roughness: 0.5
                metallic: 0.5
            },
            // we can also include other files
            // this is a textual inclusion just like C/C++ preprocessor
            @include("materials/wood.akari")
        ]
        render_pipeline: RenderPipeline {
            enable_gi: true
        }
    }
}

```