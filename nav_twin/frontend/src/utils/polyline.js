
import polyline from '@mapbox/polyline';
export function decodePolyline(encoded){
  try{ return polyline.decode(encoded).map(([lat,lng])=>({ latitude: lat, longitude: lng })); }
  catch{ return []; }
}
export function gatherCoordinates(route){
  if(route?.polyline){
    const main = decodePolyline(route.polyline);
    if(main.length) return main;
  }
  const coords = [];
  (route?.legs || []).forEach(leg=>{
    (leg.steps || []).forEach(step=>{
      const c = step.polyline ? decodePolyline(step.polyline) : [];
      if(c.length) coords.push(...c);
    });
  });
  return coords;
}
