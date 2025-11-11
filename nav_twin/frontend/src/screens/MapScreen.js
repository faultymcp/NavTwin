
import React, { useEffect, useMemo, useRef } from 'react';
import { View } from 'react-native';
import MapView, { Marker, Polyline } from 'react-native-maps';
import Button from '../components/Button'; import { gatherCoordinates } from '../utils/polyline';
export default function MapScreen({ route, navigation }){
  const { routeData } = route.params || {}; const mapRef=useRef(null); const coords=useMemo(()=>gatherCoordinates(routeData),[routeData]);
  useEffect(()=>{ if(mapRef.current && coords.length>0){ setTimeout(()=>{ mapRef.current.fitToCoordinates(coords,{edgePadding:{top:80,right:50,bottom:80,left:50},animated:true}); },300);} },[coords]);
  const start=routeData?.legs?.[0]?.start_location; const end=routeData?.legs?.slice(-1)?.[0]?.end_location;
  const initialRegion = coords.length? { latitude:coords[0].latitude, longitude:coords[0].longitude, latitudeDelta:0.05, longitudeDelta:0.05 } : { latitude:start?.lat||51.5074, longitude:start?.lng||-0.1278, latitudeDelta:0.06, longitudeDelta:0.06 };
  return (<View style={{flex:1}}><MapView ref={mapRef} style={{flex:1}} initialRegion={initialRegion}>{coords.length>0 && (<Polyline coordinates={coords} strokeWidth={5}/>) }{start && (<Marker coordinate={{latitude:start.lat,longitude:start.lng}} title="Start"/>) }{end && (<Marker coordinate={{latitude:end.lat,longitude:end.lng}} title="Destination"/>) }</MapView><View style={{position:'absolute',top:16,left:16}}><Button title="Back" variant="ghost" onPress={()=>navigation.goBack()}/></View></View>);
}
