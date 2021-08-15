//! Physics pipeline structures.

use crate::data::{ComponentSet, ComponentSetMut, ComponentSetOption};
use crate::dynamics::{
    MultibodySet, RigidBodyActivation, RigidBodyChanges, RigidBodyColliders, RigidBodyDominance,
    RigidBodyHandle, RigidBodyIds, RigidBodyPosition, RigidBodyType, RigidBodyVelocity,
};
use crate::geometry::{
    BroadPhase, BroadPhasePairEvent, ColliderBroadPhaseData, ColliderChanges, ColliderFlags,
    ColliderHandle, ColliderMaterial, ColliderPair, ColliderParent, ColliderPosition,
    ColliderShape, ColliderType, NarrowPhase,
};
use crate::math::Real;
use crate::pipeline::{EventHandler, PhysicsHooks};

#[cfg(feature = "default-sets")]
use crate::{dynamics::RigidBodySet, geometry::ColliderSet};

/// The collision pipeline, responsible for performing collision detection between colliders.
///
/// This structure only contains temporary data buffers. It can be dropped and replaced by a fresh
/// copy at any time. For performance reasons it is recommended to reuse the same physics pipeline
/// instance to benefit from the cached data.
// NOTE: this contains only workspace data, so there is no point in making this serializable.
pub struct CollisionPipeline {
    broadphase_collider_pairs: Vec<ColliderPair>,
    broad_phase_events: Vec<BroadPhasePairEvent>,
}

#[allow(dead_code)]
fn check_pipeline_send_sync() {
    fn do_test<T: Sync>() {}
    do_test::<CollisionPipeline>();
}

impl CollisionPipeline {
    /// Initializes a new physics pipeline.
    pub fn new() -> CollisionPipeline {
        CollisionPipeline {
            broadphase_collider_pairs: Vec::new(),
            broad_phase_events: Vec::new(),
        }
    }

    fn detect_collisions<Bodies, Colliders>(
        &mut self,
        prediction_distance: Real,
        broad_phase: &mut BroadPhase,
        narrow_phase: &mut NarrowPhase,
        bodies: &mut Bodies,
        colliders: &mut Colliders,
        multibodies: &MultibodySet,
        modified_colliders: &[ColliderHandle],
        removed_colliders: &[ColliderHandle],
        hooks: &dyn PhysicsHooks<Bodies, Colliders>,
        events: &dyn EventHandler,
        handle_user_changes: bool,
    ) where
        Bodies: ComponentSetMut<RigidBodyActivation>
            + ComponentSet<RigidBodyType>
            + ComponentSetMut<RigidBodyIds>
            + ComponentSet<RigidBodyDominance>,
        Colliders: ComponentSetMut<ColliderBroadPhaseData>
            + ComponentSet<ColliderChanges>
            + ComponentSet<ColliderPosition>
            + ComponentSet<ColliderShape>
            + ComponentSetOption<ColliderParent>
            + ComponentSet<ColliderType>
            + ComponentSet<ColliderMaterial>
            + ComponentSet<ColliderFlags>,
    {
        // Update broad-phase.
        self.broad_phase_events.clear();
        self.broadphase_collider_pairs.clear();

        broad_phase.update(
            prediction_distance,
            colliders,
            modified_colliders,
            removed_colliders,
            &mut self.broad_phase_events,
        );

        // Update narrow-phase.
        if handle_user_changes {
            narrow_phase.handle_user_changes(
                None,
                modified_colliders,
                removed_colliders,
                colliders,
                bodies,
                events,
            );
        }

        narrow_phase.register_pairs(None, colliders, bodies, &self.broad_phase_events, events);
        narrow_phase.compute_contacts(
            prediction_distance,
            bodies,
            colliders,
            multibodies,
            modified_colliders,
            hooks,
            events,
        );
        narrow_phase.compute_intersections(bodies, colliders, modified_colliders, hooks, events);
    }

    fn clear_modified_colliders(
        &mut self,
        colliders: &mut impl ComponentSetMut<ColliderChanges>,
        modified_colliders: &mut Vec<ColliderHandle>,
    ) {
        for handle in modified_colliders.drain(..) {
            colliders.set_internal(handle.0, ColliderChanges::empty())
        }
    }

    /// Executes one step of the collision detection.
    #[cfg(feature = "default-sets")]
    pub fn step(
        &mut self,
        prediction_distance: Real,
        broad_phase: &mut BroadPhase,
        narrow_phase: &mut NarrowPhase,
        bodies: &mut RigidBodySet,
        multibodies: &MultibodySet,
        colliders: &mut ColliderSet,
        hooks: &dyn PhysicsHooks<RigidBodySet, ColliderSet>,
        events: &dyn EventHandler,
    ) {
        let mut modified_bodies = bodies.take_modified();
        let mut modified_colliders = colliders.take_modified();
        let mut removed_colliders = colliders.take_removed();

        self.step_generic(
            prediction_distance,
            broad_phase,
            narrow_phase,
            bodies,
            multibodies,
            colliders,
            &mut modified_bodies,
            &mut modified_colliders,
            &mut removed_colliders,
            hooks,
            events,
        );
    }

    /// Executes one step of the collision detection.
    pub fn step_generic<Bodies, Colliders>(
        &mut self,
        prediction_distance: Real,
        broad_phase: &mut BroadPhase,
        narrow_phase: &mut NarrowPhase,
        bodies: &mut Bodies,
        multibodies: &MultibodySet,
        colliders: &mut Colliders,
        modified_bodies: &mut Vec<RigidBodyHandle>,
        modified_colliders: &mut Vec<ColliderHandle>,
        removed_colliders: &mut Vec<ColliderHandle>,
        hooks: &dyn PhysicsHooks<Bodies, Colliders>,
        events: &dyn EventHandler,
    ) where
        Bodies: ComponentSetMut<RigidBodyPosition>
            + ComponentSetMut<RigidBodyVelocity>
            + ComponentSetMut<RigidBodyIds>
            + ComponentSetMut<RigidBodyActivation>
            + ComponentSetMut<RigidBodyChanges>
            + ComponentSet<RigidBodyColliders>
            + ComponentSet<RigidBodyDominance>
            + ComponentSet<RigidBodyType>,
        Colliders: ComponentSetMut<ColliderBroadPhaseData>
            + ComponentSetMut<ColliderChanges>
            + ComponentSetMut<ColliderPosition>
            + ComponentSet<ColliderShape>
            + ComponentSetOption<ColliderParent>
            + ComponentSet<ColliderType>
            + ComponentSet<ColliderMaterial>
            + ComponentSet<ColliderFlags>,
    {
        super::user_changes::handle_user_changes_to_colliders(
            bodies,
            colliders,
            &modified_colliders[..],
        );
        super::user_changes::handle_user_changes_to_rigid_bodies(
            None,
            bodies,
            colliders,
            &modified_bodies,
            modified_colliders,
        );
        self.detect_collisions(
            prediction_distance,
            broad_phase,
            narrow_phase,
            bodies,
            colliders,
            multibodies,
            &modified_colliders[..],
            removed_colliders,
            hooks,
            events,
            true,
        );

        self.clear_modified_colliders(colliders, modified_colliders);
        removed_colliders.clear();
    }
}
