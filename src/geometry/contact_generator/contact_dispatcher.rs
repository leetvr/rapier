use crate::geometry::contact_generator::{
    ContactGenerator, ContactPhase, HeightFieldShapeContactGeneratorWorkspace,
    PfmPfmContactManifoldGeneratorWorkspace, PrimitiveContactGenerator,
    TrimeshShapeContactGeneratorWorkspace,
};
use crate::geometry::{Shape, ShapeType};
use std::any::Any;

/// Trait implemented by structures responsible for selecting a collision-detection algorithm
/// for a given pair of shapes.
pub trait ContactDispatcher {
    /// Select the collision-detection algorithm for the given pair of primitive shapes.
    fn dispatch_primitives(
        &self,
        shape1: ShapeType,
        shape2: ShapeType,
    ) -> (
        PrimitiveContactGenerator,
        Option<Box<dyn Any + Send + Sync>>,
    );
    /// Select the collision-detection algorithm for the given pair of non-primitive shapes.
    fn dispatch(
        &self,
        shape1: ShapeType,
        shape2: ShapeType,
    ) -> (ContactPhase, Option<Box<dyn Any + Send + Sync>>);
}

/// The default contact dispatcher used by Rapier.
pub struct DefaultContactDispatcher;

impl ContactDispatcher for DefaultContactDispatcher {
    fn dispatch_primitives(
        &self,
        shape1: ShapeType,
        shape2: ShapeType,
    ) -> (
        PrimitiveContactGenerator,
        Option<Box<dyn Any + Send + Sync>>,
    ) {
        match (shape1, shape2) {
            (ShapeType::Ball, ShapeType::Ball) => (
                PrimitiveContactGenerator {
                    generate_contacts: super::generate_contacts_ball_ball,
                    #[cfg(feature = "simd-is-enabled")]
                    generate_contacts_simd: super::generate_contacts_ball_ball_simd,
                    ..PrimitiveContactGenerator::default()
                },
                None,
            ),
            (ShapeType::Cuboid, ShapeType::Cuboid) => (
                PrimitiveContactGenerator {
                    generate_contacts: super::generate_contacts_cuboid_cuboid,
                    ..PrimitiveContactGenerator::default()
                },
                None,
            ),
            // (ShapeType::Polygon, ShapeType::Polygon) => (
            //     PrimitiveContactGenerator {
            //         generate_contacts: super::generate_contacts_polygon_polygon,
            //         ..PrimitiveContactGenerator::default()
            //     },
            //     None,
            // ),
            (ShapeType::Capsule, ShapeType::Capsule) => (
                PrimitiveContactGenerator {
                    generate_contacts: super::generate_contacts_capsule_capsule,
                    ..PrimitiveContactGenerator::default()
                },
                None,
            ),
            (_, ShapeType::Ball) | (ShapeType::Ball, _) => (
                PrimitiveContactGenerator {
                    generate_contacts: super::generate_contacts_ball_convex,
                    ..PrimitiveContactGenerator::default()
                },
                None,
            ),
            (ShapeType::Capsule, ShapeType::Cuboid) | (ShapeType::Cuboid, ShapeType::Capsule) => (
                PrimitiveContactGenerator {
                    generate_contacts: super::generate_contacts_cuboid_capsule,
                    ..PrimitiveContactGenerator::default()
                },
                None,
            ),
            (ShapeType::Triangle, ShapeType::Cuboid) | (ShapeType::Cuboid, ShapeType::Triangle) => {
                (
                    PrimitiveContactGenerator {
                        generate_contacts: super::generate_contacts_cuboid_triangle,
                        ..PrimitiveContactGenerator::default()
                    },
                    None,
                )
            }
            (ShapeType::Cylinder, _)
            | (_, ShapeType::Cylinder)
            | (ShapeType::Cone, _)
            | (_, ShapeType::Cone)
            | (ShapeType::RoundedCylinder, _)
            | (_, ShapeType::RoundedCylinder) => (
                PrimitiveContactGenerator {
                    generate_contacts: super::generate_contacts_pfm_pfm,
                    ..PrimitiveContactGenerator::default()
                },
                Some(Box::new(PfmPfmContactManifoldGeneratorWorkspace::default())),
            ),
            _ => (PrimitiveContactGenerator::default(), None),
        }
    }

    fn dispatch(
        &self,
        shape1: ShapeType,
        shape2: ShapeType,
    ) -> (ContactPhase, Option<Box<dyn Any + Send + Sync>>) {
        match (shape1, shape2) {
            (ShapeType::Trimesh, _) | (_, ShapeType::Trimesh) => (
                ContactPhase::NearPhase(ContactGenerator {
                    generate_contacts: super::generate_contacts_trimesh_shape,
                    ..ContactGenerator::default()
                }),
                Some(Box::new(TrimeshShapeContactGeneratorWorkspace::new())),
            ),
            (ShapeType::HeightField, _) | (_, ShapeType::HeightField) => (
                ContactPhase::NearPhase(ContactGenerator {
                    generate_contacts: super::generate_contacts_heightfield_shape,
                    ..ContactGenerator::default()
                }),
                Some(Box::new(HeightFieldShapeContactGeneratorWorkspace::new())),
            ),
            _ => {
                let (gen, workspace) = self.dispatch_primitives(shape1, shape2);
                (ContactPhase::ExactPhase(gen), workspace)
            }
        }
    }
}
